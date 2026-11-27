

from final_project.Morphizer import Morphizer

m = Morphizer()

def test(word: str):
  encoded = m.splitWord(word)
  decoded = m.recombineWord(encoded)
  print(word, '->', encoded, '->', decoded)
  assert(word == decoded)

test("organizationally")
test("curved")
test("happiness")
