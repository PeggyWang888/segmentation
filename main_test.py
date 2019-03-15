from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = unet()
testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,16,verbose=1)
saveResult("data/membrane/test",results)
