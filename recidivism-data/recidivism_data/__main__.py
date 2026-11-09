from . import DATA, TABLES, build

build(force=True)
print(f"wrote {', '.join(TABLES)} to {DATA}")
