import pickle

test=[1,2,3,4,5]
with open('{}.t'.format("Test"), 'wb') as nf:
    pickle.dump(test, nf)
    nf.close()
with open('{}.t'.format("Test"), 'rb') as nf:
    t = pickle.load(nf)
    nf.close()
    print(t)

