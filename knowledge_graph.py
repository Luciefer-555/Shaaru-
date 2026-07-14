"""
knowledge_graph.py — Neo4j fashion knowledge graph interface.

Connects to Neo4j AuraDB and provides Cypher query helpers for
fashion context: style pairings, aesthetic associations, brand data.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.kg")


class KnowledgeGraph:
    """Singleton wrapper around Neo4j driver with fashion-specific queries."""

    def __init__(self):
        self.driver = None
        self._connected = False
        self._connect()

    def _connect(self):
        """Attempt Neo4j connection using env vars."""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not password:
            log.info("[KG] Neo4j env vars not set — running without graph")
            return

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password), max_connection_lifetime=180, keep_alive=True
            )
            # Test connectivity
            with self.driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            log.info(f"[OK] Connected to Neo4j Knowledge Graph ({uri}).")
        except Exception as e:
            log.warning(f"[KG] Neo4j connection failed: {e}")
            self.driver = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.driver is not None

    def query(self, cypher: str, params: dict = None) -> list:
        """
        Run a Cypher query and return results as list of dicts.

        Args:
            cypher: Cypher query string.
            params: Query parameters dict.

        Returns:
            List of record dicts, or empty list on failure.
        """
        if not self.is_connected:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            log.warning(f"[KG] Query failed: {e}")
            return []

    def query_pairings(self, aesthetics: list) -> list:
        """Find trending style pairings for given aesthetics."""
        cypher = """
        MATCH (a:Aesthetic)-[:PAIRS_WITH|COMPLEMENTS]->(b:Aesthetic)
        WHERE a.name IN $aesthetics
        RETURN a.name AS base, b.name AS pair, 1.0 AS score
        ORDER BY score DESC
        LIMIT 10
        """
        return self.query(cypher, {"aesthetics": aesthetics})

    def query_item_pairings(self, item_name: str) -> list:
        """Find pairing suggestions for a specific item across Aesthetic and Construction nodes."""
        cypher = """
        MATCH (base)-[r:PAIRS_WITH|COMPLEMENTS]-(pair)
        WHERE (base:Aesthetic OR base:Construction OR base:Product)
          AND (toLower(base.name) CONTAINS toLower($item_name) OR toLower($item_name) CONTAINS toLower(base.name))
          AND (r.pairing_confidence IS NULL OR r.pairing_confidence >= 0.5)
          AND (r.evidence_status IS NULL OR r.evidence_status <> 'llm_generated_unverified')
        RETURN pair.name AS paired_item, "style pairing" AS technique, coalesce(r.pairing_confidence, 1.0) AS score
        LIMIT 10
        """
        return self.query(cypher, {"item_name": item_name})

    def query_occasion_suitability(self, occasion: str, item_labels: list = None) -> list:
        """Find Aesthetic, Construction, and Product nodes that suit a specific occasion or match scanned item labels."""
        if not self.is_connected or not occasion:
            return []
        cypher = """
        MATCH (base)-[r:SUITS_OCCASION]->(occ:Occasion)
        WITH base, occ, r,
             (toLower(occ.name) CONTAINS toLower($occasion) OR toLower($occasion) CONTAINS toLower(occ.name)) AS occasion_matched
        WHERE (base:Product OR base:Aesthetic OR base:Construction)
          AND (r.evidence_status IS NULL OR r.evidence_status <> 'llm_generated_unverified')
          AND (
            occasion_matched
            OR (
              $item_labels IS NOT NULL AND size($item_labels) > 0 AND
              any(lbl IN $item_labels WHERE lbl <> '' AND (
                toLower(coalesce(base.name, base.title, '')) CONTAINS toLower(lbl) OR
                toLower(lbl) CONTAINS toLower(coalesce(base.name, base.title, '')) OR
                (base.category IS NOT NULL AND toLower(base.category) CONTAINS toLower(lbl)) OR
                (base.garment_class IS NOT NULL AND toLower(base.garment_class) CONTAINS toLower(lbl))
              ))
            )
          )
        RETURN DISTINCT labels(base)[0] AS type, coalesce(base.title, base.name) AS item_name, occ.name AS occasion, coalesce(base.description, base.caption, '') AS description, occasion_matched
        ORDER BY occasion_matched DESC
        LIMIT 15
        """
        return self.query(cypher, {"occasion": occasion, "item_labels": item_labels or []})

    def query_construction_pairings(self, construction_name: str, gender: str = None) -> list:
        """Find pairing suggestions between Construction nodes with confidence ranking and gender filtering."""
        cypher = """
        MATCH (c1:Construction)-[r:PAIRS_WITH]-(c2:Construction)
        WHERE (toLower(c1.name) CONTAINS toLower($name) OR toLower($name) CONTAINS toLower(c1.name))
          AND (r.pairing_confidence IS NULL OR r.pairing_confidence >= 0.5)
          AND (r.evidence_status IS NULL OR r.evidence_status <> 'llm_generated_unverified')
          AND ($gender IS NULL OR c2.gender_association IS NULL OR c2.gender_association = 'unisex' OR c2.gender_association = $gender)
        RETURN c2.name AS paired_item, c2.garment_class AS garment_class, coalesce(r.pairing_confidence, 1.0) AS pairing_confidence, coalesce(r.source_urls, []) AS source_urls, coalesce(r.context_label, 'standard') AS context_label, r.evidence_status AS evidence_status
        ORDER BY pairing_confidence DESC
        LIMIT 15
        """
        return self.query(cypher, {"name": construction_name, "gender": gender})

    def query_construction_fabrics(self, construction_name: str) -> list:
        """Find required or recommended fabrics for a Construction node."""
        cypher = """
        MATCH (c:Construction)-[:REQUIRES_FABRIC]->(f:Fabric)
        WHERE toLower(c.name) CONTAINS toLower($name) OR toLower($name) CONTAINS toLower(c.name)
        RETURN c.name AS construction, f.name AS fabric
        LIMIT 10
        """
        return self.query(cypher, {"name": construction_name})

    def query_silhouette_constructions(self, aesthetics: list) -> list:
        """Find Construction silhouettes that suit specific aesthetics."""
        if not self.is_connected or not aesthetics:
            return []
        cypher = """
        MATCH (a:Aesthetic)-[:SUITS_SILHOUETTE]->(c:Construction)
        WHERE a.name IN $aesthetics
        RETURN a.name AS aesthetic, c.name AS construction, c.garment_class AS garment_class
        LIMIT 15
        """
        return self.query(cypher, {"aesthetics": aesthetics})

    def query_influencer_picks(self, limit: int = 3) -> list:
        """Get recent influencer style picks."""
        cypher = """
        MATCH (b:Brand)-[:EMBODIES]->(a:Aesthetic)
        RETURN b.name AS influencer, a.name AS item, "everyday" AS occasion
        LIMIT $limit
        """
        return self.query(cypher, {"limit": limit})

    def get_fashion_context(self, item: str) -> str:
        """
        Get fashion knowledge context for a specific item.
        Returns formatted string for brain context injection.
        """
        if not self.is_connected:
            return ""

        # Find related aesthetics, occasions, and brands
        cypher = """
        MATCH (b:Brand)
        WHERE toLower(b.name) CONTAINS toLower($item)
        OPTIONAL MATCH (b)-[:EMBODIES]->(a:Aesthetic)
        RETURN b.name AS garment,
               collect(DISTINCT a.name) AS aesthetics,
               [] AS occasions,
               [b.name] AS brands
        LIMIT 5
        """
        results = self.query(cypher, {"item": item})
        if not results:
            return ""

        lines = []
        for r in results:
            parts = [f"{r.get('garment', item)}:"]
            if r.get("aesthetics"):
                parts.append(f"aesthetics={', '.join(r['aesthetics'])}")
            if r.get("occasions"):
                parts.append(f"occasions={', '.join(r['occasions'])}")
            if r.get("brands"):
                parts.append(f"brands={', '.join(r['brands'])}")
            lines.append(" ".join(parts))

        return "\n".join(lines)

    def get_style_graph_context(self, user_id: str, aesthetics: list) -> str:
        """
        Find connected styles/items for a user's aesthetic profile.
        Returns formatted context string.
        """
        if not self.is_connected or not aesthetics:
            return ""

        cypher = """
        MATCH (a:Aesthetic)
        WHERE a.name IN $aesthetics
        OPTIONAL MATCH (a)-[:PAIRS_WITH|COMPLEMENTS]->(pair:Aesthetic)
        RETURN a.name AS aesthetic, [] AS items,
               collect(DISTINCT pair.name)[..3] AS pairs
        LIMIT 5
        """
        results = self.query(cypher, {"aesthetics": aesthetics})
        if not results:
            return ""

        lines = []
        for r in results:
            aesthetic = r.get("aesthetic", "?")
            items = r.get("items", [])
            pairs = r.get("pairs", [])
            line = f"{aesthetic}: {', '.join(items)}"
            if pairs:
                line += f" | pairs with: {', '.join(pairs)}"
            lines.append(line)

        return "Style graph:\n" + "\n".join(lines)

    def query_color_pairings(self, colors: list) -> list:
        """Find complementary and clashing colors for given colors."""
        if not self.is_connected or not colors:
            return []
        cypher = """
        MATCH (c:Color)-[r:COMPLEMENTS|CLASHES_WITH]->(x:Color)
        WHERE toLower(c.name) IN $colors
        RETURN c.name AS base_color, type(r) AS rel_type, x.name AS target_color
        LIMIT 15
        """
        return self.query(cypher, {"colors": [str(c).lower() for c in colors]})

    def query_fabric_requirements(self, aesthetics: list) -> list:
        """Find required/associated fabrics for given aesthetics."""
        if not self.is_connected or not aesthetics:
            return []
        cypher = """
        MATCH (a:Aesthetic)-[:REQUIRES_FABRIC]->(f:Fabric)
        WHERE a.name IN $aesthetics
        RETURN a.name AS aesthetic, f.name AS fabric
        LIMIT 15
        """
        return self.query(cypher, {"aesthetics": aesthetics})

    def seed_aesthetic_edges(self) -> None:
        """
        Seed PAIRS_WITH, CONFLICTS_WITH, and REQUIRES_FABRIC edges
        between the 49 existing Aesthetic nodes.
        Safe to re-run — uses MERGE, not CREATE.
        """
        pairs = [
            ("Quiet Luxury",        "Minimalist"),
            ("Quiet Luxury",        "Global Indian Chic"),
            ("Quiet Luxury",        "Old Money"),
            ("Global Indian Chic",  "Heritage Luxury"),
            ("Global Indian Chic",  "Indo Western Fusion"),
            ("Cottagecore",         "Bohemian"),
            ("Editorial",           "Avant-garde"),
            ("Streetwear",          "Minimalist"),
            ("Minimalist",          "Old Money"),
        ]
        conflicts = [
            ("Quiet Luxury",   "Maximalist Bridal"),
            ("Minimalist",     "Maximalist Bridal"),
            ("Cottagecore",    "Streetwear"),
            ("Editorial",      "Heritage Luxury"),
        ]
        fabric_requirements = [
            ("Quiet Luxury",        "raw_silk_dupion"),
            ("Quiet Luxury",        "handloom_cotton"),
            ("Quiet Luxury",        "fine_wool"),
            ("Global Indian Chic",  "chanderi_silk"),
            ("Global Indian Chic",  "ikat"),
            ("Heritage Luxury",     "banarasi_silk"),
            ("Heritage Luxury",     "kanjivaram"),
            ("Cottagecore",         "khadi_cotton"),
            ("Cottagecore",         "linen"),
            ("Streetwear",          "denim"),
            ("Streetwear",          "jersey_knit"),
            ("Minimalist",          "linen"),
            ("Minimalist",          "tencel"),
            ("Maximalist Bridal",   "heavy_silk_brocade"),
            ("Maximalist Bridal",   "velvet"),
        ]

        with self.driver.session() as session:
            for a, b in pairs:
                session.run(
                    "MATCH (a:Aesthetic {name:$a}) MATCH (b:Aesthetic {name:$b}) "
                    "MERGE (a)-[:PAIRS_WITH]->(b)",
                    a=a, b=b
                )
            for a, b in conflicts:
                session.run(
                    "MATCH (a:Aesthetic {name:$a}) MATCH (b:Aesthetic {name:$b}) "
                    "MERGE (a)-[:CONFLICTS_WITH]->(b)",
                    a=a, b=b
                )
            for aesthetic, fabric in fabric_requirements:
                session.run(
                    "MATCH (a:Aesthetic {name:$aesthetic}) "
                    "MERGE (f:Fabric {name:$fabric}) "
                    "MERGE (a)-[:REQUIRES_FABRIC]->(f)",
                    aesthetic=aesthetic, fabric=fabric
                )

        log.info("Aesthetic edges seeded — PAIRS_WITH, CONFLICTS_WITH, REQUIRES_FABRIC")

    def seed_construction_nodes_and_edges(self) -> None:
        """
        Seed 36 Construction nodes mirroring Fashionpedia and Indian ethnic categories,
        plus SUITS_SILHOUETTE, PAIRS_WITH, and REQUIRES_FABRIC relationships.
        Safe to re-run — uses MERGE, not CREATE.
        """
        if not self.is_connected:
            return

        constructions = [
            {"name": "shirt_blouse", "garment_class": "top", "silhouette": "structured/draped", "gender_association": "unisex"},
            {"name": "top_t_shirt_sweatshirt", "garment_class": "top", "silhouette": "relaxed", "gender_association": "unisex"},
            {"name": "sweater", "garment_class": "top", "silhouette": "relaxed", "gender_association": "unisex"},
            {"name": "cardigan", "garment_class": "top", "silhouette": "layered", "gender_association": "unisex"},
            {"name": "jacket", "garment_class": "outerwear", "silhouette": "structured", "gender_association": "unisex"},
            {"name": "vest", "garment_class": "outerwear", "silhouette": "layered", "gender_association": "unisex"},
            {"name": "pants", "garment_class": "bottom", "silhouette": "tailored/relaxed", "gender_association": "unisex"},
            {"name": "shorts", "garment_class": "bottom", "silhouette": "relaxed", "gender_association": "unisex"},
            {"name": "skirt", "garment_class": "bottom", "silhouette": "flowing/structured", "gender_association": "womenswear"},
            {"name": "coat", "garment_class": "outerwear", "silhouette": "structured/long", "gender_association": "unisex"},
            {"name": "dress", "garment_class": "dress", "silhouette": "one-piece", "gender_association": "womenswear"},
            {"name": "jumpsuit", "garment_class": "dress", "silhouette": "one-piece tailored", "gender_association": "womenswear"},
            {"name": "cape", "garment_class": "outerwear", "silhouette": "draped/dramatic", "gender_association": "unisex"},
            {"name": "glasses", "garment_class": "accessory", "silhouette": "accessory", "gender_association": "unisex"},
            {"name": "hat", "garment_class": "accessory", "silhouette": "headwear", "gender_association": "unisex"},
            {"name": "headband_hair_accessory", "garment_class": "accessory", "silhouette": "headwear", "gender_association": "womenswear"},
            {"name": "tie", "garment_class": "accessory", "silhouette": "neckwear", "gender_association": "menswear"},
            {"name": "glove", "garment_class": "accessory", "silhouette": "handwear", "gender_association": "unisex"},
            {"name": "watch", "garment_class": "accessory", "silhouette": "wristwear", "gender_association": "unisex"},
            {"name": "belt", "garment_class": "accessory", "silhouette": "waistwear", "gender_association": "unisex"},
            {"name": "leg_warmer", "garment_class": "accessory", "silhouette": "legwear", "gender_association": "unisex"},
            {"name": "tights_stockings", "garment_class": "accessory", "silhouette": "legwear", "gender_association": "womenswear"},
            {"name": "sock", "garment_class": "accessory", "silhouette": "footwear accessory", "gender_association": "unisex"},
            {"name": "shoe", "garment_class": "footwear", "silhouette": "footwear", "gender_association": "unisex"},
            {"name": "bag_wallet", "garment_class": "accessory", "silhouette": "carry", "gender_association": "unisex"},
            {"name": "scarf", "garment_class": "accessory", "silhouette": "draped", "gender_association": "unisex"},
            {"name": "umbrella", "garment_class": "accessory", "silhouette": "carry accessory", "gender_association": "unisex"},
            {"name": "saree", "garment_class": "dress", "silhouette": "draped", "gender_association": "womenswear"},
            {"name": "lehenga_set", "garment_class": "set", "silhouette": "voluminous skirt + cropped blouse + drape", "gender_association": "womenswear"},
            {"name": "kurta", "garment_class": "top", "silhouette": "straight/A-line tunic", "gender_association": "unisex"},
            {"name": "salwar_kameez_set", "garment_class": "set", "silhouette": "tunic + pleated trousers + drape", "gender_association": "womenswear"},
            {"name": "sharara_set", "garment_class": "set", "silhouette": "flared wide-leg pants + short tunic + drape", "gender_association": "womenswear"},
            {"name": "anarkali_dress", "garment_class": "dress", "silhouette": "frock-style flared tunic", "gender_association": "womenswear"},
            {"name": "dupatta", "garment_class": "accessory", "silhouette": "stole/drape", "gender_association": "womenswear"},
            {"name": "co_ord_set", "garment_class": "set", "silhouette": "matching top + bottom", "gender_association": "unisex"},
            {"name": "romper", "garment_class": "dress", "silhouette": "one-piece short", "gender_association": "womenswear"}
        ]

        silhouette_suits = [
            ("Quiet Luxury", "shirt_blouse"), ("Quiet Luxury", "pants"), ("Quiet Luxury", "jacket"), ("Quiet Luxury", "dress"), ("Quiet Luxury", "coat"),
            ("Minimalist", "shirt_blouse"), ("Minimalist", "pants"), ("Minimalist", "dress"), ("Minimalist", "sweater"), ("Minimalist", "vest"),
            ("Global Indian Chic", "saree"), ("Global Indian Chic", "lehenga_set"), ("Global Indian Chic", "kurta"), ("Global Indian Chic", "salwar_kameez_set"), ("Global Indian Chic", "sharara_set"), ("Global Indian Chic", "anarkali_dress"), ("Global Indian Chic", "dupatta"), ("Global Indian Chic", "co_ord_set"),
            ("Heritage Luxury", "saree"), ("Heritage Luxury", "lehenga_set"), ("Heritage Luxury", "anarkali_dress"),
            ("Indo Western Fusion", "co_ord_set"), ("Indo Western Fusion", "cape"), ("Indo Western Fusion", "jumpsuit"), ("Indo Western Fusion", "sharara_set"), ("Indo Western Fusion", "vest"),
            ("Streetwear", "top_t_shirt_sweatshirt"), ("Streetwear", "pants"), ("Streetwear", "shorts"), ("Streetwear", "jacket"), ("Streetwear", "hat"), ("Streetwear", "shoe"), ("Streetwear", "belt"),
            ("Editorial", "dress"), ("Editorial", "coat"), ("Editorial", "cape"), ("Editorial", "jumpsuit"), ("Editorial", "skirt"),
            ("Old Money", "shirt_blouse"), ("Old Money", "cardigan"), ("Old Money", "sweater"), ("Old Money", "pants"), ("Old Money", "jacket"), ("Old Money", "watch"), ("Old Money", "tie"), ("Old Money", "glove"),
            ("Cottagecore", "dress"), ("Cottagecore", "skirt"), ("Cottagecore", "cardigan"), ("Cottagecore", "headband_hair_accessory"), ("Cottagecore", "scarf")
        ]

        pairs = [
            ("shirt_blouse", "pants"), ("shirt_blouse", "skirt"), ("shirt_blouse", "jacket"), ("shirt_blouse", "vest"), ("shirt_blouse", "belt"),
            ("top_t_shirt_sweatshirt", "pants"), ("top_t_shirt_sweatshirt", "shorts"), ("top_t_shirt_sweatshirt", "jacket"), ("top_t_shirt_sweatshirt", "shoe"),
            ("sweater", "pants"), ("sweater", "skirt"), ("sweater", "coat"), ("sweater", "scarf"),
            ("cardigan", "shirt_blouse"), ("cardigan", "top_t_shirt_sweatshirt"), ("cardigan", "dress"), ("cardigan", "pants"),
            ("jacket", "shirt_blouse"), ("jacket", "pants"), ("jacket", "skirt"), ("jacket", "dress"), ("jacket", "vest"),
            ("kurta", "pants"), ("kurta", "dupatta"), ("kurta", "jacket"), ("kurta", "scarf"),
            ("saree", "shirt_blouse"), ("saree", "dupatta"), ("saree", "bag_wallet"), ("saree", "shoe"),
            ("lehenga_set", "dupatta"), ("lehenga_set", "bag_wallet"), ("lehenga_set", "shoe"), ("lehenga_set", "headband_hair_accessory"),
            ("salwar_kameez_set", "dupatta"), ("salwar_kameez_set", "shoe"), ("salwar_kameez_set", "bag_wallet"),
            ("sharara_set", "dupatta"), ("sharara_set", "shoe"), ("sharara_set", "bag_wallet"),
            ("anarkali_dress", "dupatta"), ("anarkali_dress", "shoe"), ("anarkali_dress", "bag_wallet"),
            ("dress", "jacket"), ("dress", "coat"), ("dress", "cardigan"), ("dress", "belt"), ("dress", "shoe"), ("dress", "bag_wallet"),
            ("jumpsuit", "jacket"), ("jumpsuit", "belt"), ("jumpsuit", "shoe"), ("jumpsuit", "bag_wallet"),
            ("co_ord_set", "jacket"), ("co_ord_set", "coat"), ("co_ord_set", "shoe"), ("co_ord_set", "bag_wallet"), ("co_ord_set", "glasses")
        ]

        fab_reqs = [
            ("shirt_blouse", "cotton_poplin"), ("shirt_blouse", "cotton_broadcloth"), ("shirt_blouse", "linen_plain"),
            ("top_t_shirt_sweatshirt", "cotton_jersey"), ("top_t_shirt_sweatshirt", "french_terry"), ("top_t_shirt_sweatshirt", "modal_jersey"),
            ("sweater", "merino_knit"), ("sweater", "cashmere_plain"),
            ("jacket", "wool_gabardine"), ("jacket", "poly_viscose_suiting_twill"), ("jacket", "tweed_wool"),
            ("pants", "wool_gabardine"), ("pants", "cotton_twill"), ("pants", "denim_raw_selvedge"),
            ("dress", "silk_crepe_de_chine"), ("dress", "satin_silk"), ("dress", "georgette_polyester"),
            ("saree", "silk_charmeuse"), ("saree", "shantung_silk"), ("saree", "georgette_polyester"), ("saree", "brocade_polyester"),
            ("lehenga_set", "silk_velvet"), ("lehenga_set", "raw_silk_dupion"), ("lehenga_set", "brocade_polyester"),
            ("kurta", "cotton_poplin"), ("kurta", "belgian_linen"), ("kurta", "raw_silk_dupion"),
            ("salwar_kameez_set", "cotton_poplin"), ("salwar_kameez_set", "georgette_polyester"), ("salwar_kameez_set", "silk_crepe_de_chine"),
            ("sharara_set", "georgette_polyester"), ("sharara_set", "silk_velvet"), ("sharara_set", "brocade_polyester"),
            ("anarkali_dress", "georgette_polyester"), ("anarkali_dress", "silk_crepe_de_chine"), ("anarkali_dress", "raw_silk_dupion"),
            ("dupatta", "georgette_polyester"), ("dupatta", "silk_charmeuse"), ("dupatta", "brocade_polyester"),
            ("co_ord_set", "belgian_linen"), ("co_ord_set", "cotton_poplin"), ("co_ord_set", "poly_viscose_suiting_twill")
        ]

        with self.driver.session() as session:
            for item in constructions:
                session.run(
                    "MERGE (c:Construction {name: $name}) "
                    "ON CREATE SET c.garment_class = $garment_class, c.silhouette = $silhouette, c.gender_association = $gender_association "
                    "ON MATCH SET c.garment_class = $garment_class, c.silhouette = $silhouette, c.gender_association = $gender_association",
                    name=item["name"], garment_class=item["garment_class"], silhouette=item["silhouette"], gender_association=item["gender_association"]
                )
            for aes, con in silhouette_suits:
                session.run(
                    "MATCH (a:Aesthetic {name: $aes}) MATCH (c:Construction {name: $con}) "
                    "MERGE (a)-[:SUITS_SILHOUETTE]->(c)",
                    aes=aes, con=con
                )
            for c1, c2 in pairs:
                session.run(
                    "MATCH (a:Construction {name: $c1}) MATCH (b:Construction {name: $c2}) "
                    "MERGE (a)-[:PAIRS_WITH]->(b)",
                    c1=c1, c2=c2
                )
            for con, fab in fab_reqs:
                session.run(
                    "MATCH (c:Construction {name: $con}) "
                    "MERGE (f:Fabric {name: $fab}) "
                    "MERGE (c)-[:REQUIRES_FABRIC]->(f)",
                    con=con, fab=fab
                )

        log.info("Construction nodes and edges seeded successfully")

    def sync_product_document(self, doc: dict) -> bool:
        """
        Upsert a pipeline product document into Neo4j.
        All writes use MERGE — fully idempotent, safe to re-run.
        
        Args:
            doc: Fully processed product dict from run_pipeline.py
        Returns:
            True on success, False on failure (non-blocking)
        """
        if not self.is_connected:
            log.warning("[KG] Skipping sync — Neo4j not connected")
            return False

        for attempt in range(2):
            try:
                with self.driver.session() as session:
                    session.execute_write(self._sync_product_tx, doc)
                log.info(f"[KG] Synced: {doc.get('title', 'Unknown')}")
                return True
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                    continue
                log.warning(f"[KG] Sync failed for {doc.get('title')}: {e}")
                return False

    @staticmethod
    def _sync_product_tx(tx, doc: dict):
        """All Cypher in a single transaction — atomic."""
        source_id   = str(doc.get("source_id", "") or doc.get("id", ""))
        title       = doc.get("title", "")
        designer    = doc.get("designer", "")
        source_url  = doc.get("source_url", "")
        aesthetic   = doc.get("aesthetic_category", "")
        season      = doc.get("colour_intelligence", {}).get("season_wearability", "")
        region      = doc.get("region_of_craft", "")
        occasions   = doc.get("occasion_suitability", []) or doc.get("occasion_tags", []) or []
        raw_tech = doc.get("techniques", {})
        techniques = raw_tech.get("confirmed", []) if isinstance(raw_tech, dict) else (raw_tech if isinstance(raw_tech, list) else [])
        techniques = [t.lower().strip() for t in techniques if isinstance(t, str) and t.strip()]

        price       = str(doc.get("price", "") or "")
        images      = doc.get("images", [])
        image_url   = images[0] if (isinstance(images, list) and images and isinstance(images[0], str)) else doc.get("image_url", "")
        cap_obj     = doc.get("caption", {})
        caption     = cap_obj.get("text", "") if isinstance(cap_obj, dict) else str(cap_obj or doc.get("raw_description", "") or "")

        # 1. Upsert Product node
        tx.run("""
            MERGE (p:Product {source_id: $source_id})
            SET p.title      = $title,
                p.designer   = $designer,
                p.source_url = $source_url,
                p.price      = $price,
                p.image_url  = $image_url,
                p.caption    = $caption,
                p.synced_at  = datetime()
        """, source_id=source_id, title=title, designer=designer,
             source_url=source_url, price=price, image_url=image_url, caption=caption)

        # 2. Product → Aesthetic
        if aesthetic and isinstance(aesthetic, str):
            tx.run("""
                MERGE (a:Aesthetic {name: $aesthetic})
                WITH a
                MATCH (p:Product {source_id: $source_id})
                MERGE (p)-[:BELONGS_TO]->(a)
            """, aesthetic=aesthetic.strip(), source_id=source_id)

        # 3. Product → Season
        if season and isinstance(season, str):
            tx.run("""
                MERGE (s:Season {name: $season})
                WITH s
                MATCH (p:Product {source_id: $source_id})
                MERGE (p)-[:SUITS_SEASON]->(s)
            """, season=season.strip(), source_id=source_id)

        # 4. Product → Occasions (UNWIND batch)
        valid_occasions = [o.strip() for o in occasions if o and isinstance(o, str)]
        if valid_occasions:
            tx.run("""
                MATCH (p:Product {source_id: $source_id})
                UNWIND $occasions AS occ
                MERGE (o:Occasion {name: occ})
                MERGE (p)-[:SUITS_OCCASION]->(o)
            """, occasions=valid_occasions, source_id=source_id)

        # 5. Product → Techniques → Region (UNWIND batch)
        valid_techniques = [t.strip() for t in techniques if t and isinstance(t, str)]
        if valid_techniques:
            tx.run("""
                MATCH (p:Product {source_id: $source_id})
                UNWIND $techniques AS tech
                MERGE (t:Technique {name: tech})
                MERGE (p)-[:HAS_TECHNIQUE]->(t)
            """, techniques=valid_techniques, source_id=source_id)
            if region and isinstance(region, str):
                tx.run("""
                    UNWIND $techniques AS tech
                    MATCH (t:Technique {name: tech})
                    MERGE (r:Region {name: $region})
                    MERGE (t)-[:CRAFT_ORIGIN]->(r)
                """, techniques=valid_techniques, region=region.strip())

        # 6. Product → Brand node
        if designer and isinstance(designer, str):
            tx.run("""
                MERGE (b:Brand {name: $designer})
                WITH b
                MATCH (p:Product {source_id: $source_id})
                MERGE (p)-[:MADE_BY]->(b)
            """, designer=designer.strip(), source_id=source_id)

        # 7. Product → Fabric nodes
        raw_fab = doc.get("fabric_vocabulary", {})
        fabrics_list = raw_fab.get("confirmed", []) if isinstance(raw_fab, dict) else (raw_fab if isinstance(raw_fab, list) else [])
        if not fabrics_list:
            fabrics_list = [f.get("fabric_id_guess") for f in doc.get("fabrics", []) if isinstance(f, dict) and f.get("fabric_id_guess")]

        cleaned_fabrics = []
        for item in fabrics_list:
            if isinstance(item, dict):
                fid = item.get("fabric_id") or item.get("name")
                if fid and isinstance(fid, str): cleaned_fabrics.append(fid.strip())
            elif isinstance(item, str) and item.strip():
                cleaned_fabrics.append(item.strip())

        for fab in cleaned_fabrics:
            tx.run("""
                MERGE (f:Fabric {name: $fab})
                WITH f
                MATCH (p:Product {source_id: $source_id})
                MERGE (p)-[:HAS_FABRIC]->(f)
            """, fab=fab, source_id=source_id)

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            try:
                self.driver.close()
            except Exception:
                pass


# ── Singleton ────────────────────────────────────────────────────
kg = None

class _FallbackKG:
    driver = None
    is_connected = False
    def query(self, *a, **kw): return []
    def query_pairings(self, *a, **kw): return []
    def query_item_pairings(self, *a, **kw): return []
    def query_construction_pairings(self, *a, **kw): return []
    def query_construction_fabrics(self, *a, **kw): return []
    def query_silhouette_constructions(self, *a, **kw): return []
    def seed_construction_nodes_and_edges(self): pass
    def query_influencer_picks(self, *a, **kw): return []
    def get_fashion_context(self, *a, **kw): return ""
    def get_style_graph_context(self, *a, **kw): return ""
    def query_color_pairings(self, *a, **kw): return []
    def query_fabric_requirements(self, *a, **kw): return []
    def sync_product_document(self, *a, **kw): return False
    def close(self): pass

def get_kg():
    global kg
    if kg is None:
        try:
            kg = KnowledgeGraph()
        except Exception as e:
            log.warning(f"[KG] Failed to initialize: {e}")
            kg = _FallbackKG()
    return kg
