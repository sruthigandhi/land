"""
Pull real summer NDVI (June-Aug) from Earth Engine for KY counties.
Uses Landsat (since it covers 2002 onward) with cloud masking and median composite.
Takes ~15 min for all years/counties the first time.
"""
import os, time, ee
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
service_account = os.getenv("GEE_SERVICE_ACCOUNT")
key_path = os.getenv("GEE_KEY_PATH")
project = os.getenv("GEE_PROJECT")

# Initialize Earth Engine with service account
credentials = ee.ServiceAccountCredentials(service_account, key_path)
ee.Initialize(credentials, project=project, opt_url='https://earthengine-highvolume.googleapis.com')

def get_ky_counties():
    """Fetch KY counties from TIGER/2018."""
    return ee.FeatureCollection("TIGER/2018/Counties").filter(
        ee.Filter.eq("STATEFP", "21")
    )

def get_summer_ndvi_composite(year):
    """
    Cloud-masked, scaled summer NDVI composite.
    Use Landsat 7 for 2002-2012, Landsat 8 for 2013+.
    """
    start_date = f"{year}-06-01"
    end_date = f"{year}-08-31"
    
    if year < 2013:
        collection_id = "LANDSAT/LE07/C02/T1_L2"
        nir_band = "SR_B4"
        red_band = "SR_B3"
        qa_band = "QA_PIXEL"
    else:
        collection_id = "LANDSAT/LC08/C02/T1_L2"
        nir_band = "SR_B5"
        red_band = "SR_B4"
        qa_band = "QA_PIXEL"
    
    def cloud_mask(img):
        """Mask clouds and shadows using QA band."""
        qa = img.select(qa_band)
        cloud_bit = 1 << 3
        shadow_bit = 1 << 4
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
        # Scale reflectance bands
        optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
        return img.addBands(optical, None, True).updateMask(mask)
    
    def add_ndvi(img):
        """Compute NDVI = (NIR - Red) / (NIR + Red)."""
        ndvi = img.normalizedDifference([nir_band, red_band]).rename("NDVI")
        return img.addBands(ndvi)
    
    ky = get_ky_counties()
    collection = (
        ee.ImageCollection(collection_id)
        .filterDate(start_date, end_date)
        .filterBounds(ky)
        .map(cloud_mask)
        .map(add_ndvi)
        .select("NDVI")
    )
    return collection.median()

def compute_county_ndvi(year):
    """Compute mean NDVI for each KY county."""
    print(f"  Computing NDVI for {year}...", end="", flush=True)
    ky = get_ky_counties()
    ndvi_image = get_summer_ndvi_composite(year)
    
    # Reduce to county means
    reduced = ndvi_image.reduceRegions(
        collection=ky,
        reducer=ee.Reducer.mean(),
        scale=30
    )
    
    # Export to list (small enough for getInfo)
    features = reduced.getInfo()["features"]
    rows = []
    for f in features:
        props = f["properties"]
        rows.append({
            "county_fips": props.get("GEOID"),
            "county": props.get("NAME", "").upper(),
            "year": year,
            "ndvi": props.get("mean")
        })
    print(f" {len([r for r in rows if r['ndvi'] is not None])} counties with data")
    return rows

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    all_rows = []
    for year in [2002, 2007, 2012, 2017, 2022]:
        rows = compute_county_ndvi(year)
        all_rows.extend(rows)
        time.sleep(1)  # Avoid rate limits
    
    df = pd.DataFrame(all_rows)
    df = df[df["ndvi"].notna()]
    df.to_csv("data/ndvi_raw.csv", index=False)
    print(f"\nSaved {len(df)} county-year NDVI observations to data/ndvi_raw.csv")
    print(df.pivot_table(index="year", aggfunc="size"))
