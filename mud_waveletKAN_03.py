import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import networkx as nx
import bz2
import xml.etree.ElementTree as ET
from collections import namedtuple
import random
from scipy.sparse import vstack
from wikidump_reader_2 import WikipediaDumpReader
import argparse

# --- Collecting and Visualising --
class ProgressVisualizer:
    """A class to generate visualizations before and after training."""
    def __init__(self, embeddings, graph):
        """Initializes with the data needed for pre-training visuals."""
        self.embeddings = embeddings
        self.graph = graph
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_embedding_clusters(self, n_clusters=8, save_path="embedding_clusters.png"):
        """Clusters embeddings, plots them in 2D, and calculates silhouette score."""
        print(f"Running K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(self.embeddings)
        labels = kmeans.labels_
        
        score = silhouette_score(self.embeddings, labels)
        print(f"Silhouette Score: {score:.4f}")

        print("Reducing dimensions with PCA for plotting...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.title(f'Initial Embedding Clusters (Silhouette Score: {score:.4f})', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(handles=scatter.legend_elements()[0], labels=range(n_clusters), title="Clusters")
        plt.savefig(save_path)
        plt.close()

    def plot_knn_graph_sample(self, n_samples=75, save_path="knn_graph_sample.png"):
        """Visualizes a small, random sample of the k-NN graph."""
        if self.graph.number_of_nodes() < n_samples:
            n_samples = self.graph.number_of_nodes()

        nodes = np.random.choice(self.graph.nodes(), n_samples, replace=False)
        subgraph = self.graph.subgraph(nodes)
        
        sample_embeddings = self.embeddings[nodes]
        pca = PCA(n_components=2)
        pos = pca.fit_transform(sample_embeddings)
        pos = {node: p for node, p in zip(nodes, pos)}

        plt.figure(figsize=(14, 14))
        nx.draw(subgraph, pos, with_labels=False, node_size=50, width=0.5, alpha=0.8, node_color='skyblue')
        plt.title(f'Sample of the k-NN Knowledge Graph ({n_samples} nodes)', fontsize=16)
        plt.savefig(save_path)
        plt.close()
        
    def generate_pre_training_visuals(self):
        """Runs all visualization tasks that can be done before training."""
        print("\n Generating pre-training visualizations...")
        self.plot_embedding_clusters()
        self.plot_knn_graph_sample()
        print("Pre-training visualizations saved as PNG files.")

    @staticmethod
    def plot_loss_curve(loss_history, save_path="loss_curve.png"):
        """Plots the training loss over epochs. Can be called independently after training."""
        print("\n Generating post-training visualization...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b')
        plt.title('Training Loss Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(range(1, len(loss_history) + 1))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print("Loss curve visualization saved as PNG file.")

# --- Model and Training Classes ---

class WaveletKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, wavelet='db4', scales=5):
        super().__init__()
        self.wavelet = wavelet
        self.scales = scales
        # Initialize basis matrices with a size guaranteed to be large enough.
        # The coefficient length at any level will not exceed the original input dimension.
        self.basis = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, output_dim))
            for s in range(scales)
        ])

    def forward(self, x):
        # Detach the tensor from the computation graph before converting to NumPy.
        coeffs = pywt.wavedec(x.cpu().detach().numpy(), self.wavelet, level=self.scales-1, axis=-1)
        transformed = []

        for s in range(self.scales):
            # Convert the numpy coefficients to a torch tensor
            coeff_s_tensor = torch.tensor(coeffs[s], device=x.device, dtype=torch.float32)

            # Get the actual length of the coefficients at the current scale
            level_len = coeff_s_tensor.shape[1]

            # Slice the oversized basis matrix to match the actual coefficient length.
            basis = self.basis[s][:level_len, :]

            # Perform the matrix multiplication
            transformed.append(torch.matmul(coeff_s_tensor, basis))

        return torch.sum(torch.stack(transformed), dim=0)

class DendriticBlock(nn.Module):
    def __init__(self, in_dim, out_dim, branches):
        super().__init__()
        self.branches = branches
        self.dendrites = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())
            for _ in range(branches)
        ])

    def forward(self, x):
        branch_outputs = [d(x) for d in self.dendrites]
        return torch.max(torch.stack(branch_outputs), dim=0)[0]

class MengerWavKAN(nn.Module):
    def __init__(self, bert_dim=768, latent_dim=256, branches=20):
        super().__init__()
        self.encoder = nn.Sequential(
            WaveletKANLayer(bert_dim, 512),
            nn.LayerNorm(512),
            DendriticBlock(512, 256, branches)
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            DendriticBlock(latent_dim, 512, branches),
            WaveletKANLayer(512, bert_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        encoded = self.encoder(x)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class WikipediaKG:
    def __init__(self, index_path, dump_path):
        self.wiki_reader = WikipediaDumpReader(index_path, dump_path) # 
        self.scaler = MinMaxScaler()

    def build_graph(self, n_articles=10000, k=5, batch_size=1024,device='cpu',inference_batch_size=32):
        """
        Constructs the graph with a focus on minimizing peak memory usage.
        BERT is loaded and released locally.
        """
        articles = self.wiki_reader.get_random_sample(n_articles)
        texts = [a['text'][:512] for a in articles if a and 'text' in a and a['text']]
        print(f"\nGot {len(texts)} articles of 512, going for embeddings ...") 

        # 1. Load BERT model to device for this function only
        print(f"Loading BERT for embedding generation on device {device}")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if device == 'cuda':
            model = BertModel.from_pretrained('bert-base-uncased').to(device)
        else:
            model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()

        print(f"Batch Tokenizing with the Bert pretrained 'bert-base-uncased'")
        all_embeddings = []
        print(f"Generating embeddings in batches of {inference_batch_size}...")
        # --- NEW: Process texts in batches ---
        for i in range(0, len(texts), inference_batch_size):
            batch_texts = texts[i:i + inference_batch_size]
        
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if device == 'cuda' :
                inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
        
            # Calculate mean embedding for the batch and move to CPU
            if device == 'cuda':
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            else:
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            progress =f"Adding a batch {i}"
            print(progress,end="\r", flush=True)
            all_embeddings.append(batch_embeddings)

        # Concatenate embeddings from all batches
        embeddings = np.vstack(all_embeddings)

    
        # Move embeddings back to cpu
        if device == 'cuda' :
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        # --- Immediately release BERT model memory ---
        del model
        del tokenizer
        if device == 'cuda':
            torch.cuda.empty_cache()

        print("BERT model released from memory.")

        # Scale embeddings
        print("Scaling embeddings.")
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        del embeddings # Delete original embeddings to save space

        # 2. Build k-NN graph in batches
        print(f"Building k-NN graph in batches of {batch_size}...")
        nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine').fit(embeddings_scaled)

        adj_parts = []
        batch_num = 1
        for start_idx in range(0, embeddings_scaled.shape[0], batch_size):
            print(f"k-NN graph batch numnber {batch_num}\r")
            end_idx = min(start_idx + batch_size, embeddings_scaled.shape[0])
            adj_batch = nn_model.kneighbors_graph(embeddings_scaled[start_idx:end_idx], mode='connectivity')
            adj_parts.append(adj_batch)
        adj = vstack(adj_parts)

        # --- Immediately release NearestNeighbors model memory ---
        del nn_model
        print("NearestNeighbors model released from memory.")

        self.graph = nx.from_scipy_sparse_array(adj)
        return embeddings_scaled, adj


class LoSwarm:
    def __init__(self, model_creator, n_particles=20, social_weight=0.05, device='cpu'):
        """
        Initializes the swarm, moving each model to the specified device.

        Args:
            model_creator (function): A function that creates a new model instance.
            n_particles (int): The number of models (particles) in the swarm.
            social_weight (float): The influence of the best model on other particles.
            device (str): The device to run the models on ('cuda' or 'cpu').
        """
        if device != 'cpu':
            self.particles = [model_creator().to(device) for _ in range(n_particles)]
        else:
            self.particles = [model_creator() for _ in range(n_particles)]
        self.optimizers = [torch.optim.Adam(p.parameters(), lr=1e-3) for p in self.particles]
        self.best_loss = float('inf')
        self.best_model = None
        self.social_weight = social_weight
        self.device = device

    def train_epoch(self, data, adj):
        epoch_losses = []
        for model, opt in zip(self.particles, self.optimizers):
            model.train()
            opt.zero_grad()

            recon, mu, logvar = model(data, adj)
            recon_loss = F.mse_loss(recon, data)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kld

            loss.backward()
            opt.step()

            if self.best_model is not None and model is not self.best_model:
                with torch.no_grad():
                    for param, best_param in zip(model.parameters(), self.best_model.parameters()):
                        param.data += self.social_weight * (best_param.data - param.data)

            loss_item = loss.item()
            epoch_losses.append(loss_item)

            if loss_item < self.best_loss:
                self.best_loss = loss_item
                self.best_model = model

        return np.mean(epoch_losses)

class KnowledgeGenerator:
    def __init__(self, model, wiki_kg, device='cpu'):
        self.model = model
        self.wiki_kg = wiki_kg # Provides the scale
        self.device = device

        # This generator is used for the final step, so it loads its own BERT model.
        print(f"\nKnowledgeGenerator: Loading BERT model for text generation onto device {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def generate(self, prompt=None, max_length=50, temp=0.7):
        self.model.eval()
        with torch.no_grad():
            if prompt:
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
                inputs = {key: val.to(self.device) for key, val in inputs.items()} # Move prompt to device

                outputs = self.bert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                emb = self.wiki_kg.scaler.transform(emb)

                encoded = self.model.encoder(torch.FloatTensor(emb).to(self.device))
                mu, _ = self.model.mu(encoded), self.model.logvar(encoded)
                z = mu
            else:
                z = torch.randn(1, self.model.mu.out_features,device=self.device)

            gen_emb = self.model.decoder(z)
            gen_emb = gen_emb.cpu().numpy()
            gen_emb = self.wiki_kg.scaler.inverse_transform(gen_emb)

            sim = torch.cosine_similarity(
                torch.FloatTensor(gen_emb).to(self.device),
                self.bert.embeddings.word_embeddings.weight,
                dim=-1
            )
            tokens = []
            for _ in range(max_length):
                probs = F.softmax(sim / temp, -1)
                token_id = torch.multinomial(probs, 1).item()
                tokens.append(self.tokenizer.decode([token_id]))
                next_emb = self.bert.embeddings.word_embeddings.weight[token_id].unsqueeze(0)
                sim = torch.cosine_similarity(next_emb, self.bert.embeddings.word_embeddings.weight, dim=-1)

            return ' '.join(tokens).replace(' ##', '')

# --- Usage Example ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MengerWavKAN model on Wikipedia data.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Number of epochs to train with the swarm.")
    parser.add_argument('--articles', type=int, default=10000,
                        help='Number of Wikipedia articles to process.')
    parser.add_argument('--particles', type=int, default=20,
                        help='Number of particles (models) in the swarm.')
    parser.add_argument('--social_weight', type=float, default=0.05,
                        help='Influence of the best model on other particles in the swarm.')
    parser.add_argument('--device',default='cpu',
                        help='For setting the preferred target')
    args = parser.parse_args()
    # 1. Initialize Wikipedia reader and build graph
    print(f"Initializing Wikipedia Knowledge Graph with {args.articles} articles...")
    wiki = WikipediaKG(
        index_path="data/wiki/enwiki-20241201-pages-articles-multistream-index.txt.bz2",
        dump_path="data/wiki/enwiki-20241201-pages-articles-multistream.xml.bz2"
        )
    embeddings_np, adj_matrix = wiki.build_graph(n_articles=args.articles, k=5)

    # 2. Convert data to PyTorch tensors
    print("Converting data to tensors...")
    adj_coo = adj_matrix.tocoo()
    adj = torch.sparse_coo_tensor(
        np.vstack([adj_coo.row, adj_coo.col]),
        adj_coo.data.astype(np.float32),
        adj_coo.shape
    )
    data = torch.FloatTensor(embeddings_np)

    # Process numpy array with the visualiser ---
    visualiser = ProgressVisualizer(embeddings_np, wiki.graph)
    visualiser.generate_pre_training_visuals()

    # --- Clean up the large numpy array and visualiser---
    del embeddings_np
    del visualiser
    print("Numpy embeddings array and viualiser released from memory.")

    # 3. Initialize swarm of models
    print(f"\nInitializing model swarm with {args.particles} particles...")
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device="cpu"
    print(f"Using device: {device}")
    adj = adj.to(device)
    data = data.to(device)
    def create_model():
        return MengerWavKAN(bert_dim=768, latent_dim=256, branches=20)

    swarm = LoSwarm(create_model,
                    n_particles=args.particles,
                    social_weight=args.social_weight,
                    device=device)
    
    # 4. Training loop
    print("Starting training...")
    loss_history = []
    for epoch in range(args.epochs): #? Reduced epochs for demonstration
        loss = swarm.train_epoch(data, adj)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Best Loss: {swarm.best_loss:.4f}")
        loss_history.append(loss)

    # 5. Generate from the best model
    print("\n--- Generating Knowledge ---")
    if swarm.best_model:
        generator = KnowledgeGenerator(swarm.best_model, wiki)

        # Prompt-based generation
        print("\n--- Prompt-based Generation ---")
        prompt = "Quantum entanglement in physics"
        print(f"Prompt: '{prompt}'")
        print(generator.generate(prompt, max_length=100))

        # Random generation
        print("\n--- Random Generation ---")
        print(generator.generate(max_length=100))
    else:
        print("Training did not result in a best model. Cannot generate text.")

    ProgressVisualizer.plot_loss_curve(loss_history)

    print("\n Finished") 



