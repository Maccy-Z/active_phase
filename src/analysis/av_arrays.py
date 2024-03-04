import numpy as np

# Ours
xs = \
[
[0.13850415512465375, 0.23268698060941828, 0.4986149584487535, 0.13019390581717452, 0.07479224376731301, 0.0997229916897507, 0.10526315789473684, 0.10249307479224377, 0.060941828254847646, 0.04986149584487535, 0.036011080332409975, 0.038781163434903045, 0.038781163434903045, 0.0443213296398892, 0.0443213296398892, 0.038781163434903045, 0.036011080332409975, 0.036011080332409975, 0.0332409972299169, 0.030470914127423823, 0.027700831024930747, 0.027700831024930747, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.01662049861495845, 0.0221606648199446, 0.019390581717451522, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.019390581717451522, 0.013850415512465374, 0.013850415512465374, 0.0110803324099723, 0.01662049861495845, 0.01662049861495845, 0.01662049861495845, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.01662049861495845, 0.013850415512465374, 0.01662049861495845, 0.01662049861495845],
[0.13850415512465375, 0.23268698060941828, 0.4986149584487535, 0.13019390581717452, 0.10526315789473684, 0.12742382271468145, 0.10249307479224377, 0.11080332409972299, 0.06925207756232687, 0.0664819944598338, 0.055401662049861494, 0.055401662049861494, 0.05817174515235457, 0.05817174515235457, 0.05263157894736842, 0.04709141274238227, 0.04709141274238227, 0.05263157894736842, 0.055401662049861494, 0.04986149584487535, 0.036011080332409975, 0.027700831024930747, 0.030470914127423823, 0.027700831024930747, 0.027700831024930747, 0.027700831024930747, 0.01662049861495845, 0.01662049861495845, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.013850415512465374, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.0221606648199446, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.01662049861495845, 0.01662049861495845, 0.01662049861495845, 0.013850415512465374, 0.01662049861495845, 0.01662049861495845],
[0.13850415512465375, 0.17174515235457063, 0.1329639889196676, 0.10249307479224377, 0.13573407202216067, 0.12465373961218837, 0.1772853185595568, 0.15512465373961218, 0.11080332409972299, 0.11357340720221606, 0.09141274238227147, 0.08033240997229917, 0.0664819944598338, 0.060941828254847646, 0.05817174515235457, 0.05817174515235457, 0.055401662049861494, 0.055401662049861494, 0.038781163434903045, 0.04709141274238227, 0.04709141274238227, 0.04709141274238227, 0.04709141274238227, 0.04986149584487535, 0.04986149584487535, 0.04709141274238227, 0.055401662049861494, 0.05263157894736842, 0.05263157894736842, 0.04709141274238227, 0.04709141274238227, 0.04709141274238227, 0.04986149584487535, 0.04986149584487535, 0.04155124653739612, 0.04155124653739612, 0.04155124653739612, 0.036011080332409975, 0.036011080332409975, 0.030470914127423823, 0.030470914127423823, 0.027700831024930747, 0.027700831024930747, 0.024930747922437674, 0.024930747922437674, 0.024930747922437674, 0.024930747922437674, 0.0221606648199446, 0.024930747922437674, 0.024930747922437674, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446],
[0.13850415512465375, 0.13573407202216067, 0.1772853185595568, 0.11357340720221606, 0.12465373961218837, 0.16066481994459833, 0.19113573407202217, 0.1440443213296399, 0.09141274238227147, 0.06925207756232687, 0.08587257617728532, 0.06371191135734072, 0.07202216066481995, 0.07479224376731301, 0.07479224376731301, 0.07479224376731301, 0.0664819944598338, 0.04709141274238227, 0.04709141274238227, 0.036011080332409975, 0.038781163434903045, 0.038781163434903045, 0.038781163434903045, 0.036011080332409975, 0.036011080332409975, 0.04155124653739612, 0.036011080332409975, 0.0332409972299169, 0.0332409972299169, 0.024930747922437674, 0.024930747922437674, 0.0332409972299169, 0.0332409972299169, 0.024930747922437674, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.01662049861495845, 0.01662049861495845, 0.01662049861495845, 0.01662049861495845, 0.01662049861495845, 0.01662049861495845, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723, 0.0110803324099723],
[0.13850415512465375, 0.23268698060941828, 0.32409972299168976, 0.10803324099722991, 0.11357340720221606, 0.12188365650969529, 0.14681440443213298, 0.09418282548476455, 0.10526315789473684, 0.09141274238227147, 0.07479224376731301, 0.07756232686980609, 0.07756232686980609, 0.06371191135734072, 0.05817174515235457, 0.05817174515235457, 0.05817174515235457, 0.04709141274238227, 0.038781163434903045, 0.038781163434903045, 0.038781163434903045, 0.038781163434903045, 0.038781163434903045, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.0332409972299169, 0.030470914127423823, 0.027700831024930747, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.030470914127423823, 0.027700831024930747, 0.0221606648199446, 0.0221606648199446, 0.024930747922437674, 0.024930747922437674, 0.024930747922437674, 0.024930747922437674, 0.024930747922437674, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522, 0.019390581717451522],
[0.13850415512465375, 0.14681440443213298, 0.1883656509695291, 0.13850415512465375, 0.13573407202216067, 0.13850415512465375, 0.13573407202216067, 0.09695290858725762, 0.11080332409972299, 0.10526315789473684, 0.08033240997229917, 0.08033240997229917, 0.07479224376731301, 0.0664819944598338, 0.0443213296398892, 0.0443213296398892, 0.04709141274238227, 0.04709141274238227, 0.05263157894736842, 0.04155124653739612, 0.04155124653739612, 0.04155124653739612, 0.04155124653739612, 0.038781163434903045, 0.038781163434903045, 0.038781163434903045, 0.036011080332409975, 0.036011080332409975, 0.024930747922437674, 0.027700831024930747, 0.027700831024930747, 0.013850415512465374, 0.013850415512465374, 0.019390581717451522, 0.01662049861495845, 0.0221606648199446, 0.024930747922437674, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.019390581717451522, 0.019390581717451522, 0.0221606648199446, 0.024930747922437674, 0.019390581717451522, 0.0221606648199446, 0.024930747922437674, 0.027700831024930747, 0.027700831024930747]
]

xs = np.array(xs)

for i in [10,20,30,40,50]:
    print(np.mean(xs[:, i]))

print()
mean = np.mean(xs, axis=0)

m = [str(a) + "," for a in mean]
m = "".join(m)
print("[", m,"]")

std = np.std(xs, axis=0)
m = [str(a) + "," for a in std]
m = "".join(m)
print("[", m,"]")
