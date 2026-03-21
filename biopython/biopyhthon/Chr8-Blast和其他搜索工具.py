from Bio import motifs
from Bio.Seq import Seq
instances = [Seq("TACAA"),
Seq("TACGC"),
Seq("TACAC"),
Seq("TACCC"),
Seq("AACCC"),
Seq("AATGC"),
Seq("AATGC")]
m = motifs.create(instances)
print(m)
print(m.counts)
m.weblogo("mymotif.png")

from Bio import LogisticRegression
xs = [[-53, -200.78],
[117, -267.14],
[57, -163.47],
[16, -190.30],
[11, -220.94],
[85, -193.94],
[16, -182.71],
[15, -180.41],
[-26, -181.73],
[58, -259.87],
[126, -414.53],
[191, -249.57],
[113, -265.28],
[145, -312.99],
[154, -213.83],
[147, -380.85],[93, -291.13]]
ys = [1,1,1,1, 1,1,1,1,
1,
1,
0,
0,
0,
0,
0,
0,
0]
model = LogisticRegression.train(xs, ys)
print(model)
print(model.beta)
def show_progress(iteration, loglikelihood):
    print("Iteration:", iteration, "Log-likelihood function:", loglikelihood)
model1 = LogisticRegression.train(xs, ys, update_fn=show_progress)
print(model1)
from Bio import kNN
k = 3
model = kNN.train(xs, ys, k)
print(model)