import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
from wikipedia2vec import Wikipedia2Vec
import networkx as nx

class WaveletKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, wavelet='db4', scales=5):
        super().__init__()
        self.wavelet = wavelet
        self.scales = scales
        self.basis = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim // (2**s), output_dim))
            for s in range(scales)
        ])
        
    def forward(self, x):
        coeffs = pywt.wavedec(x.cpu().numpy(), self.wavelet, level=self.scales-1, axis=-1)
        transformed = []
        for s in range(self.scales):
            level = coeffs[s].shape[-1]
            basis = self.basis[s][:level]
            transformed.append(torch.matmul(torch.tensor(coeffs[s], device=x.device), basis))
        return torch.sum(torch.stack(transformed), dim=0)

class MengerWavKAN(nn.Module):
    def __init__(self, bert_dim=768, latent_dim=256, branches=20):
        super().__init__()
        # Menger-inspired dendritic encoder
        self.encoder = nn.Sequential(
            WaveletKANLayer(bert_dim, 512),
            nn.LayerNorm(512),
            DendriticBlock(512, 256, branches)
        )
        
        # Latent space projections
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        
        # Menger decoder with parallel branches
        self.decoder = nn.Sequential(
            DendriticBlock(latent_dim, 512, branches),
            WaveletKANLayer(512, bert_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, adj):
        # Graph aggregation
        x = torch.sparse.mm(adj, x)
        encoded = self.encoder(x)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class DendriticBlock(nn.Module):
    def __init__(self, in_dim, out_dim, branches):
        super().__init__()
        self.branches = branches
        self.dendrites = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU()
            ) for _ in range(branches)
        ])
        
    def forward(self, x):
        branch_outputs = [d(x) for d in self.dendrites]
        return torch.max(torch.stack(branch_outputs), dim=0)[0]

class WikipediaKG:
    def __init__(self, dump_path):
        self.wiki = Wikipedia2Vec.load(dump_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.scaler = MinMaxScaler()
        
    def build_graph(self, n_articles=10000, k=5):
        """Construct semantic graph from Wikipedia articles"""
        articles = [self.wiki.get_article(str(i)) for i in range(n_articles)]
        texts = [a.text[:512] for a in articles]
        
        # Generate BERT embeddings
        inputs = self.tokenizer(texts, return_tensors='pt', 
                              padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        # Normalize and build KNN graph
        self.embeddings = self.scaler.fit_transform(embeddings)
        adj = kneighbors_graph(self.embeddings, n_neighbors=k, mode='connectivity')
        self.graph = nx.from_scipy_sparse_array(adj)
        return self.embeddings, adj

class LoSwarm:
    def __init__(self, model, n_particles=20):
        self.particles = [model() for _ in range(n_particles)]
        self.optimizers = [torch.optim.Adam(p.parameters(), lr=1e-3) 
                         for p in self.particles]
        
    def train_epoch(self, data, adj):
        losses = []
        for model, opt in zip(self.particles, self.optimizers):
            model.train()
            opt.zero_grad()
            
            recon, mu, logvar = model(data, adj)
            recon_loss = F.mse_loss(recon, data)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1*kld
            
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return np.mean(losses)

class KnowledgeGenerator:
    def __init__(self, model, wiki_kg):
        self.model = model
        self.wiki_kg = wiki_kg
        self.bert = wiki_kg.bert
        self.tokenizer = wiki_kg.tokenizer
        
    def generate(self, prompt=None, max_length=50, temp=0.7):
        self.model.eval()
        with torch.no_grad():
            if prompt:
                inputs = self.tokenizer(prompt, return_tensors='pt',
                                      padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.bert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).numpy()
                emb = self.wiki_kg.scaler.transform(emb)
                mu, _ = self.model.encoder(torch.FloatTensor(emb))
                z = mu
            else:
                z = torch.randn(1, self.model.mu.out_features)
                
            gen_emb = self.model.decoder(z)
            gen_emb = self.wiki_kg.scaler.inverse_transform(gen_emb.numpy())
            
            # Find closest tokens in BERT vocabulary
            sim = torch.cosine_similarity(
                torch.FloatTensor(gen_emb),
                self.bert.embeddings.word_embeddings.weight,
                dim=-1
            )
            tokens = []
            for _ in range(max_length):
                probs = F.softmax(sim/temp, -1)
                token_id = torch.multinomial(probs, 1).item()
                tokens.append(self.tokenizer.decode([token_id]))
            return ' '.join(tokens).replace(' ##', '')

# Usage
if __name__ == "__main__":
    # 1. Load Wikipedia data
    wiki = WikipediaKG("enwiki_dump.bin")
    embeddings, adj = wiki.build_graph(n_articles=10000)
    
    # 2. Convert to PyTorch tensors
    adj = adj.tocoo()
    adj = torch.sparse_coo_tensor(
        np.vstack([adj.row, adj.col]),
        adj.data,
        adj.shape
    )
    data = torch.FloatTensor(embeddings)
    
    # 3. Initialize swarm of models
    def create_model():
        return MengerWavKAN(bert_dim=768, latent_dim=256, branches=20)
    
    swarm = LoSwarm(create_model, n_particles=10)
    
    # 4. Training loop
    for epoch in range(100):
        loss = swarm.train_epoch(data, adj)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 5. Generate from best model
    best_model = min(swarm.particles, key=lambda m: m.training_loss)
    generator = KnowledgeGenerator(best_model, wiki)
    
    # Prompt-based generation
    print(generator.generate("Quantum entanglement in", max_length=100))
    
    # Random generation
    print(generator.generate(max_length=100))

