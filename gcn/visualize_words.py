from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

import sys
import os

if len(sys.argv) != 2:
    sys.exit("Use: python build_graph.py <dataset>")

dataset_name = sys.argv[1]

word_vector_dir = os.path.join(
    "data", "{}_word_vectors.txt".format(dataset_name))

f = open(word_vector_dir, 'r')
embedding_lines = f.readlines()
f.close()

target_names = set()
labels = []
docs = []
for i in range(len(embedding_lines)):
    line = embedding_lines[i].strip()
    temp = line.split('\t')
    emb_str = embedding_lines[i].strip().split()
    values_str_list = emb_str[1:]
    values = [float(x) for x in values_str_list]
    label = np.argmax(values)
    docs.append(values)
    target_names.add(label)
    labels.append(label)

target_names = list(target_names)

label = np.array(labels)

fea = TSNE(n_components=2).fit_transform(docs)
if not os.path.exists("results"):
    os.mkdir("results")

pdf_dir = os.path.join("results", "{}_gcn_word_test.pdf".format(dataset_name))
pdf = PdfPages(pdf_dir)
cls = np.unique(label)

# cls=range(10)
fea_num = [fea[label == i] for i in cls]
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i])
# plt.legend(ncol=2,  )
# plt.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.48, -0.08),fontsize=11)
# plt.ylim([-20,35])
# plt.title(md_file)
plt.tight_layout()
pdf.savefig()
plt.show()
pdf.close()
