Created At: 2026-06-12T20:49:35Z
Completed At: 2026-06-12T20:49:35Z
File Path: `file:///C:/Users/saipr/Downloads/Shaaru/api.py`
Total Lines: 1421
Total Bytes: 66051
Showing lines 1263 to 1330
The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
1263: @app.get("/api/products/seed")
1264: async def seed_products(
1265:     user_id: str = Query("test_user", description="User ID"),
1266:     aesthetic: str = Query(None, description="Comma-separated aesthetics"),
1267:     category: str = Query(None, description="Comma-separated categories"),
1268:     query: str = Query(None, description="Search query"),
1269:     _token: dict = Depends(verify_token)
1270: ):
1271:     seed_path = Path("products_seed.json")
1272:     if not seed_path.exists():
1273:         return {"products": [], "total": 0}
1274:         
1275:     try:
1276:         with open(seed_path, "r", encoding="utf-8") as f:
1277:             data = json.load(f)
1278:     except Exception as e:
1279:         print(f"Error reading seed file: {e}")
1280:         return {"products": [], "total": 0}
1281:         
1282:     products = data.get("products", [])
1283:     
1284:     # 2. CATEGORY FILTER (BEFORE SCORING)
1285:     if category and category.lower() != "view all":
1286:         cat_lower = category.lower()
1287:         everyday = ["shirts", "t-shirts", "graphic tees", "polos", "chinos", "trousers", "jeans", "casual pants", "shorts"]
1288:         cozy = ["hoodies", "sweatshirts", "sweatpants", "joggers", "loungewear", "knitwear", "sweaters", "pullovers", "oversized", "fleece"]
1289:         editorial = ["blazers", "jackets", "coats", "trench", "waistcoat", "structured", "formal", "statement", "leather jacket", "bomber", "windbreaker"]
1290:         accessories = ["shoes", "sneakers", "boots", "loafers", "sunglasses", "belts", "caps", "hat
<truncated 71 bytes>
gs", "ties", "pocket squares"]
1291:         
1292:         target_cats = []
1293:         if "everyday" in cat_lower: target_cats = everyday
1294:         elif "cozy" in cat_lower: target_cats = cozy
1295:         elif "editorial" in cat_lower: target_cats = editorial
1296:         elif "accessories" in cat_lower: target_cats = accessories
1297:         
1298:         if target_cats:
1299:             filtered = []
1300:             for p in products:
1301:                 p_cat = p.get("category", "").lower()
1302:                 if any(tc in p_cat for tc in target_cats):
1303:                     filtered.append(p)
1304:             products = filtered
1305:             
1306:     # SEARCH
1307:     if query:
1308:         q = query.lower()
1309:         products = [p for p in products if q in p.get("product_name", "").lower() or q in p.get("description", "").lower() or q in p.get("brand", "").lower()]
1310: 
1311:     # SCORING
1312:     profile = db["comfort_profiles"].find_one({"user_id": user_id}) or {}
1313:     
1314:     def score_product(product, profile):
1315:         score = 0
1316:         user_aesthetics = (
1317:             profile.get('aesthetics', []) + 
1318:             profile.get('style_potential', {}).get('best_aesthetics', [])
1319:         )
1320:         
1321:         product_aesthetics = product.get('aesthetic', [])
1322:         for ua in user_aesthetics:
1323:             for pa in product_aesthetics:
1324:                 if ua.lower() in pa.lower() or pa.lower() in ua.lower():
1325:                     score += 3
1326:         
1327:         critical_gaps = profile.get('gaps', {}).get('critical_gaps', [])
1328:         for gap in critical_gaps:
1329:             if isinstance(gap, dict):
1330:                 missing = gap.get('missing_piece', '').lower()
The above content does NOT show the entire file contents. If you need to view any lines of the file which were not shown to complete your task, call this tool again to view those lines.
