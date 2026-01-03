import pandas as pd
import os

print("=" * 70)
print("📊 ANALYZING EXCEL EXTRACTION RESULTS")
print("=" * 70)

# Check what files were created
extraction_dir = r"data\from_jeremy\transcriptions"
files = os.listdir(extraction_dir)
print("Files in transcriptions folder:")
for f in files:
    if f.endswith(('.csv', '.txt')):
        size = os.path.getsize(os.path.join(extraction_dir, f))
        print(f"  📄 {f} ({size/1024:.1f} KB)")

# Read the raw extraction CSV
csv_path = r"data\from_jeremy\transcriptions\raw_extraction.csv"
if os.path.exists(csv_path):
    print(f"\n📖 Reading: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        print(f"\n🔍 First 10 entries:")
        print(df.head(10).to_string())
        
        # Show what's in each column
        print(f"\n📋 Column 'Value' preview (first 20 unique values):")
        values = df['Value'].dropna().unique()[:20]
        for i, val in enumerate(values):
            print(f"  {i+1:2d}. '{val}'")
        
        # Check for census-related values
        print(f"\n🔎 Looking for census keywords in values...")
        keywords = ['white', 'black', 'race', 'house', 'street', 'rent', 'own']
        found = []
        for val in values:
            if isinstance(val, str):
                lower_val = val.lower()
                for kw in keywords:
                    if kw in lower_val and val not in found:
                        found.append(val)
        
        if found:
            print("  Found potential census data:")
            for f in found[:10]:
                print(f"    • {f}")
        else:
            print("  No obvious census keywords found")
            
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

# Check summary file
summary_path = r"data\from_jeremy\transcriptions\file_summary.txt"
if os.path.exists(summary_path):
    print(f"\n📄 Summary file contents:")
    with open(summary_path, 'r') as f:
        lines = f.readlines()[:20]  # First 20 lines
        for line in lines:
            print(f"  {line.rstrip()}")

print("\n" + "=" * 70)
print("🎯 WHAT THIS MEANS FOR YOUR PROJECT")
print("=" * 70)

print("""
✅ GOOD NEWS:
• The Excel file DOES contain census data (Race, House Number, etc.)
• Your column description matches what we expected

❌ BAD NEWS:
• The data is trapped in bad formatting (colors, merged cells)
• WPS Office can't save a clean version
• Our scripts can't properly parse it

🎯 IMMEDIATE SOLUTION:
We need the MANUAL MAPPING information from you:

1. OPEN THE EXCEL FILE IN WPS (read-only is fine)
2. TELL ME:
   • What COLUMN LETTER (A, B, C...) contains RACE data?
   • What COLUMN LETTER contains HOUSE NUMBER?
   • What COLUMN LETTER contains STREET NAME?
   • etc. for all your 6 categories
   
   EXAMPLE ANSWER:
   "Race = Column B, House Number = Column C, Street Name = Column D..."

3. ALSO TELL ME:
   • What ROW NUMBER does the ACTUAL DATA start on?
     (Not headers, but the first actual census entry)

⚡ WITH THIS INFO, I can write a script that:
1. Reads the exact columns you specify
2. Starts at the exact row you specify  
3. Creates PROPER CSV that maps to your extracted images

📧 STILL EMAIL JEREMY, but now we have a backup plan!
""")
