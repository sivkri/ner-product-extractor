from main import extract_storage, extract_color, extract_brand

def test_extract_storage():
    assert extract_storage("512 GB SSD") == "512 GB"
    assert extract_storage("1 TB HDD") == "1 TB"
    assert extract_storage("16 GB RAM") is None

def test_extract_color():
    assert extract_color("This is a black laptop") == "Schwarz"
    assert extract_color("Available in silver color") == "Silber"
    assert extract_color("No color mentioned") is None

def test_extract_brand():
    assert extract_brand("Dell Inspiron 15") == "Dell"
    assert extract_brand("  HP Pavilion") == "HP"
    assert extract_brand("") is None
