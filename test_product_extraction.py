import unittest
from your_script import (
    extract_storage,
    extract_color,
    extract_brand_from_headline,
    convert_storage_to_gb,
    standardize_color
)

class TestProductDetailExtraction(unittest.TestCase):

    def test_extract_storage(self):
        self.assertEqual(extract_storage("Smartphone mit 128 GB Speicher"), "128 GB")
        self.assertEqual(extract_storage("Laptop mit 1 TB Festplatte"), "1 TB")
        self.assertIsNone(extract_storage("8 GB RAM Arbeitsspeicher"))
        self.assertIsNone(extract_storage("Kein Speicher erwähnt"))

    def test_convert_storage_to_gb(self):
        self.assertEqual(convert_storage_to_gb("128 GB"), 128)
        self.assertEqual(convert_storage_to_gb("1 TB"), 1024)
        self.assertEqual(convert_storage_to_gb("2 TB + 512 GB"), 2560)
        self.assertIsNone(convert_storage_to_gb(None))

    def test_extract_color(self):
        self.assertEqual(extract_color("Farbe: Schwarz"), "Schwarz".lower())
        self.assertEqual(extract_color("Erhältlich in Dunkelblau"), "dunkelblau")
        self.assertIsNone(extract_color("Kein Farbangabe"))

    def test_extract_brand_from_headline(self):
        self.assertEqual(extract_brand_from_headline("Samsung Galaxy S21"), "Samsung")
        self.assertEqual(extract_brand_from_headline("  Apple iPhone 13"), "Apple")
        self.assertIsNone(extract_brand_from_headline(None))

    def test_standardize_color(self):
        self.assertEqual(standardize_color("schwarz"), "Schwarz")
        self.assertEqual(standardize_color("white"), "Weiß")
        self.assertEqual(standardize_color("unknown"), "unknown")
        self.assertIsNone(standardize_color(None))

if __name__ == "__main__":
    unittest.main()
