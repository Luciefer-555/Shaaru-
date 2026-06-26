import re

class SilhouetteEnforcer:
    def __init__(self):
        self.targets = {
            "lehenga": 12,
            "saree": 8,
            "anarkali": 6,
            "kurta_set": 6,
            "sherwani": 8,
            "gown": 4,
            "sharara": 4,
            "other": 2
        }
        self.counts = {k: 0 for k in self.targets}
        self.total_scraped = 0
        
    def detect_silhouette(self, title, product_type, tags):
        text = f"{title} {product_type} {' '.join(tags)}".lower()
        if re.search(r'\blehenga\b|\bcholi\b', text):
            return "lehenga"
        if re.search(r'\bsari\b|\bsaree\b', text):
            return "saree"
        if re.search(r'\banarkali\b', text):
            return "anarkali"
        if re.search(r'\bkurta\b|\bkurti\b', text):
            return "kurta_set"
        if re.search(r'\bsherwani\b', text):
            return "sherwani"
        if re.search(r'\bgown\b|\bdress\b', text):
            return "gown"
        if re.search(r'\bsharara\b|\bgharara\b', text):
            return "sharara"
        return "other"
        
    def should_accept(self, item_dict):
        self.total_scraped += 1
        title = item_dict.get("title", "")
        product_type = item_dict.get("product_type", "")
        tags = item_dict.get("tags", [])
        
        silhouette = self.detect_silhouette(title, product_type, tags)
        
        if self.counts[silhouette] < self.targets[silhouette]:
            self.counts[silhouette] += 1
            return True
            
        # REDISTRIBUTION: If we've processed a large number of products (e.g. over 150)
        # and we still are asking for more to fill the limit of 50, it means the designer
        # doesn't make enough of the remaining targets. We must redistribute by relaxing the constraint.
        if self.total_scraped > 150:
            self.counts[silhouette] += 1
            return True
            
        return False
