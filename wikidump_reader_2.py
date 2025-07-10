import bz2
import xml.etree.ElementTree as ET
from collections import namedtuple
import random
ArticleInfo = namedtuple('ArticleInfo', ['offset', 'page_id', 'title'])

class WikipediaDumpReader:
    def __init__(self, index_path, dump_path):
        self.dump_path = dump_path
        self.index = self._load_index(index_path)
        
    def _load_index(self, index_path):
        """Load and parse the multistream index file"""
        index = {}
        with bz2.open(index_path, 'rt') as f:
            for line in f:
                parts = line.strip().split(':', 2)
                if len(parts) == 3:
                    info = ArticleInfo(
                        offset=int(parts[0]),
                        page_id=int(parts[1]),
                        title=parts[2]
                    )
                    index[info.title] = info
                    index[info.page_id] = info
        return index

    def _decompress_stream(self, offset):
        """Decompress a bz2 stream from the multistream file"""
        with open(self.dump_path, 'rb') as f:
            f.seek(offset)
            decompressor = bz2.BZ2Decompressor()
            
            chunks = []
            while not decompressor.eof:
                chunk = f.read(1024*1024)  # Read 1MB chunks
                if not chunk:
                    break
                chunks.append(decompressor.decompress(chunk))
                
            return b''.join(chunks).decode('utf-8')

    def _parse_xml(self, xml_content):
        """Parse XML content and extract article text"""
        # Wrap in root element for proper XML parsing
        wrapped = f"<root>{xml_content}</root>"
        root = ET.fromstring(wrapped)
        
        articles = []
        for page in root.findall('page'):
            title = page.find('title').text
            page_id = int(page.find('id').text)
            text = page.find('.//text').text or ""
            
            articles.append({
                'title': title,
                'page_id': page_id,
                'text': text
            })
            
        return articles

    def get_article(self, identifier):
        """Get article by title or page ID"""
        info = self.index.get(identifier)
        if not info:
            return None
            
        xml_content = self._decompress_stream(info.offset)
        articles = self._parse_xml(xml_content)
        
        # Find exact match in case of multiple articles per stream
        for article in articles:
            if (article['title'] == info.title or 
                article['page_id'] == info.page_id):
                return article
                
        return None

    def get_random_sample(self, sample_size=10):
        """Get random sample of articles"""
        titles = [k for k in self.index.keys() if isinstance(k, str)]
        if sample_size > len(titles):
            sample_size = len(titles)
        
        sample_keys = random.sample(titles, sample_size)
        
        articles = []
        for k in sample_keys:
            article = self.get_article(k)
            if article:
                articles.append(article)
        return articles

