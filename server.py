def get_ebird_species_for_region(country_code):
    """Fetches official native species codes for a country from eBird API and caches in memory."""
    if not country_code or country_code == "GLOBAL":
        return None

    if country_code in REGION_SPECIES_CACHE:
        return REGION_SPECIES_CACHE[country_code]

    try:
        url = f"https://api.ebird.org/v2/product/spplist/{country_code}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            species_list = json.loads(resp.read().decode('utf-8'))
            # Store lowercased species codes
            species_set = {s.lower() for s in species_list}
            REGION_SPECIES_CACHE[country_code] = species_set
            print(f"🌍 eBird species list cached for {country_code}: {len(species_set)} native species.", flush=True)
            return species_set
    except Exception as e:
        print(f"⚠️ eBird API fetch failed for {country_code}: {e}", flush=True)
        return None
