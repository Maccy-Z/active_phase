# Plot the results
import matplotlib.pyplot as plt
import numpy as np

our = \
    [0.13850415512465375, 0.13850415512465375, 0.16897506925207753, 0.13019390581717452, 0.1191135734072022, 0.0886426592797784, 0.12188365650969528, 0.12188365650969528, 0.08310249307479224,
     0.10710987996306558, 0.09418282548476455, 0.07202216066481996, 0.06555863342566944, 0.07202216066481994, 0.060941828254847646, 0.054478301015697145, 0.05263157894736842, 0.05124653739612189,
     0.04755309325946445, 0.04339796860572484, 0.03924284395198522, 0.037396121883656507, 0.038781163434903045, 0.03324099722991689, 0.03601108033240997, 0.04339796860572484, 0.0397045244690674,
     0.03462603878116344, 0.03139427516158818, 0.03139427516158819, 0.02770083102493075, 0.026777469990766394, 0.02862419205909511, 0.024469067405355493, 0.02077562326869806, 0.023084025854108958,
     0.02585410895660203, 0.02770083102493075, 0.030009233610341645, 0.030470914127423827, 0.030009233610341645, 0.030932594644506, 0.030470914127423827, 0.03047091412742382, 0.029085872576177285,
     0.030470914127423827, 0.028162511542012932, 0.024469067405355496, 0.023084025854108958, 0.023084025854108958, 0.024007386888273318, 0.024007386888273318, ]
base = [ 0.1357,0.2036,0.2419,0.2419,0.2031,0.1717,0.1551,0.1620,0.1279,0.1274,0.1168,0.0840,0.0868,0.0771,0.0669,0.0859,0.0808,0.0799,0.0706,0.0660,0.0554,0.0702,0.0813,0.0748,0.0572,0.0669,0.0591,0.0517,0.0600,0.0494,0.0508,0.0476,0.0485,0.0425,0.0420,0.0388,0.0360,0.0379,0.0360,0.0383,0.0355,0.0406,0.0420,0.0365,0.0379,0.0420,0.0383,0.0355,0.0337,0.0328,0.0319, ]
svc = [ 0.1385,0.3638,0.2207,0.2031,0.1727,0.1348,0.1256,0.0748,0.0859,0.1320,0.0817,0.0882,0.1094,0.0808,0.0974,0.0813,0.0697,0.0619,0.0531,0.0600,0.0646,0.0536,0.0365,0.0568,0.0480,0.0471,0.0526,0.0485,0.0522,0.0416,0.0452,0.0420,0.0397,0.0416,0.0434,0.0425,0.0494,0.0466,0.0485,0.0577,0.0531,0.0512,0.0512,0.0549,0.0605,0.0499,0.0494,0.0536,0.0591,0.0545,0.0499,0.0466, ]

our_std = [ 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0091,0.0118,0.0078,0.0300,0.0207,0.0193,0.0065,0.0045,0.0014,0.0030,0.0035,0.0085,0.0067,0.0039,0.0023,0.0039,0.0057,0.0057,0.0059,0.0063,0.0026,0.0066,0.0050,0.0035,0.0034,0.0075,0.0069,0.0031,0.0032,0.0049,0.0090,0.0067,0.0082,0.0053,0.0070,0.0097,0.0104,0.0076,0.0085,0.0057,0.0061,0.0031,0.0031, ]
svc_std = [ 0.0000,0.0026,0.0026,0.0496,0.0144,0.0170,0.0183,0.0157,0.0141,0.0398,0.0101,0.0090,0.0378,0.0124,0.0487,0.0270,0.0217,0.0196,0.0081,0.0267,0.0263,0.0153,0.0090,0.0152,0.0157,0.0134,0.0077,0.0101,0.0121,0.0064,0.0177,0.0108,0.0089,0.0107,0.0069,0.0097,0.0061,0.0115,0.0120,0.0190,0.0191,0.0112,0.0131,0.0094,0.0125,0.0095,0.0090,0.0126,0.0121,0.0127,0.0083,0.0099, ]
base_std = [ 0.0000,0.0263,0.0599,0.0458,0.0458,0.0451,0.0557,0.0232,0.0159,0.0223,0.0315,0.0359,0.0360,0.0245,0.0203,0.0158,0.0149,0.0242,0.0178,0.0143,0.0190,0.0180,0.0133,0.0097,0.0073,0.0146,0.0063,0.0069,0.0195,0.0067,0.0104,0.0059,0.0083,0.0086,0.0072,0.0106,0.0090,0.0217,0.0126,0.0112,0.0095,0.0050,0.0067,0.0084,0.0076,0.0109,0.0041,0.0061,0.0122,0.0067,0.0065, ]

print(len(our), len(base), len(svc))
print(len(our_std), len(base_std), len(svc_std))
accs = [our, base, svc]
stds = [our_std, base_std, svc_std]
label = ["Ours", "GPR", "SVC"]

plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.17, bottom=0.15)
for i, (a, l) in enumerate(zip(accs, label)):
    plt.plot(a, label=l)

for i, (s, l) in enumerate(zip(stds, label)):
    plt.fill_between(np.arange(len(s)), np.array(accs[i]) - np.array(s), np.array(accs[i]) + np.array(s), alpha=0.2)


plt.legend(fontsize=12)
plt.ylim([0, 0.2])
plt.yticks(np.linspace(0., 0.2, 5), fontsize=12)

plt.ylabel("Fractional error", fontsize=12)
plt.xlabel("Number of steps", fontsize=12)
plt.xticks(np.linspace(0, 50, 6), fontsize=12)
plt.tight_layout()
plt.show()
#
# # Quad PD
#
# our = \
#     [0.5595567867036011, 0.3767313019390582, 0.32409972299168976, 0.32132963988919666, 0.25761772853185594, 0.17174515235457063, 0.1523545706371191, 0.12188365650969529, 0.19667590027700832, 0.24653739612188366, 0.1745152354570637, 0.13573407202216067, 0.15512465373961218, 0.18005540166204986, 0.1329639889196676, 0.0997229916897507, 0.13850415512465375, 0.12188365650969529, 0.10803324099722991, 0.1329639889196676, 0.10526315789473684, 0.11634349030470914, 0.13573407202216067, 0.12465373961218837, 0.14127423822714683, 0.12742382271468145, 0.11080332409972299, 0.12465373961218837, 0.11357340720221606, 0.10526315789473684, 0.08587257617728532, 0.07479224376731301, 0.06371191135734072, 0.060941828254847646, 0.060941828254847646, 0.05817174515235457, 0.06371191135734072, 0.05817174515235457, 0.05817174515235457, 0.038781163434903045, 0.05817174515235457, 0.04709141274238227, 0.05817174515235457, 0.038781163434903045, 0.027700831024930747, 0.027700831024930747, 0.030470914127423823, 0.027700831024930747, 0.030470914127423823, 0.036011080332409975, 0.0443213296398892, 0.036011080332409975, 0.0332409972299169, 0.0221606648199446, 0.0221606648199446, 0.0221606648199446, 0.0110803324099723, 0.013850415512465374, 0.0110803324099723, 0.0110803324099723, 0.01662049861495845, 0.0221606648199446, 0.013850415512465374, 0.0110803324099723, 0.008310249307479225, 0.00554016620498615, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.00554016620498615, 0.00554016620498615, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.0, 0.002770083102493075, 0.0, 0.002770083102493075, 0.0, 0.002770083102493075, 0.002770083102493075, 0.00554016620498615, 0.00554016620498615, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.002770083102493075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# svc = \
#        [0.7506925207756233, 0.3684210526315789, 0.47368421052631576, 0.3545706371191136, 0.40443213296398894, 0.3601108033240997, 0.3684210526315789, 0.29085872576177285, 0.28254847645429365, 0.25761772853185594, 0.21606648199445982, 0.24376731301939059, 0.16066481994459833, 0.16066481994459833, 0.13850415512465375, 0.17174515235457063, 0.13850415512465375, 0.13019390581717452, 0.0664819944598338, 0.0997229916897507, 0.0664819944598338, 0.07756232686980609, 0.11357340720221606, 0.11357340720221606, 0.11634349030470914, 0.10526315789473684, 0.11357340720221606, 0.12465373961218837, 0.09418282548476455, 0.11911357340720222, 0.10249307479224377, 0.09418282548476455, 0.11357340720221606, 0.13573407202216067, 0.11080332409972299, 0.0997229916897507, 0.10526315789473684, 0.10803324099722991, 0.0997229916897507, 0.10803324099722991, 0.13573407202216067, 0.10526315789473684, 0.09141274238227147, 0.0997229916897507, 0.10803324099722991, 0.11911357340720222, 0.11357340720221606, 0.13850415512465375, 0.12465373961218837, 0.09695290858725762, 0.08587257617728532, 0.10526315789473684, 0.09695290858725762, 0.09695290858725762, 0.0886426592797784, 0.06925207756232687, 0.10249307479224377, 0.09418282548476455, 0.0997229916897507, 0.09141274238227147, 0.06925207756232687, 0.055401662049861494, 0.060941828254847646, 0.055401662049861494, 0.06925207756232687, 0.08033240997229917, 0.09695290858725762, 0.11357340720221606, 0.0997229916897507, 0.09141274238227147, 0.08033240997229917, 0.08587257617728532, 0.08587257617728532, 0.05817174515235457, 0.05817174515235457, 0.060941828254847646, 0.04986149584487535, 0.08033240997229917, 0.0664819944598338, 0.07479224376731301, 0.09141274238227147, 0.10803324099722991, 0.0886426592797784, 0.07202216066481995, 0.09141274238227147, 0.0886426592797784, 0.09141274238227147, 0.07202216066481995, 0.07756232686980609, 0.07202216066481995, 0.07756232686980609, 0.07202216066481995, 0.04986149584487535, 0.04155124653739612, 0.060941828254847646, 0.060941828254847646, 0.06925207756232687, 0.06371191135734072, 0.08033240997229917, 0.06371191135734072, 0.06925207756232687]
# accs = [our, svc]
# label = ["Ours", "SVC"]
#
# print(len(our))
# print(our[80], svc[80])
#
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.17, bottom=0.15)
# for i, (a, l) in enumerate(zip(accs, label)):
#     plt.plot(a, label=l)
#
# plt.legend(fontsize=12)
# plt.ylim([0, 0.2])
# plt.yticks(np.linspace(0., 0.2, 5), fontsize=12)
#
# plt.ylabel("Fractional error", fontsize=12)
# plt.xlabel("Number of steps", fontsize=12)
# plt.xticks(np.linspace(0, 100, 6), fontsize=12)
# plt.tight_layout()
# plt.show()

# Skyrmion
# our = \
# [0.14058956916099774, 0.12244897959183673, 0.13151927437641722, 0.12698412698412698, 0.12471655328798185, 0.1292517006802721, 0.14512471655328799, 0.1292517006802721, 0.07936507936507936, 0.10204081632653061, 0.12244897959183673, 0.12018140589569161, 0.09977324263038549, 0.10657596371882086, 0.06802721088435375, 0.05442176870748299, 0.061224489795918366, 0.061224489795918366, 0.06802721088435375, 0.05215419501133787, 0.049886621315192746, 0.06575963718820861, 0.05215419501133787, 0.03854875283446712, 0.05895691609977324, 0.07256235827664399, 0.05442176870748299, 0.047619047619047616, 0.047619047619047616, 0.049886621315192746, 0.034013605442176874, 0.031746031746031744, 0.036281179138321996, 0.047619047619047616, 0.05442176870748299, 0.04081632653061224, 0.05895691609977324, 0.04308390022675737, 0.049886621315192746, 0.047619047619047616, 0.04308390022675737, 0.03854875283446712, 0.03854875283446712, 0.03854875283446712, 0.045351473922902494, 0.05215419501133787, 0.036281179138321996, 0.024943310657596373, 0.036281179138321996, 0.036281179138321996, 0.036281179138321996, 0.034013605442176874, 0.024943310657596373, 0.027210884353741496, 0.02947845804988662, 0.034013605442176874, 0.034013605442176874, 0.031746031746031744, 0.031746031746031744, 0.031746031746031744, 0.034013605442176874, 0.031746031746031744, 0.02947845804988662, 0.018140589569160998, 0.02947845804988662, 0.024943310657596373, 0.027210884353741496, 0.022675736961451247, 0.022675736961451247, 0.022675736961451247, 0.022675736961451247, 0.022675736961451247, 0.024943310657596373, 0.022675736961451247, 0.024943310657596373, 0.022675736961451247, 0.022675736961451247, 0.02040816326530612, 0.02040816326530612, 0.02040816326530612, 0.022675736961451247, 0.02040816326530612, 0.018140589569160998, 0.018140589569160998, 0.018140589569160998, 0.02040816326530612, 0.02040816326530612, 0.02040816326530612, 0.013605442176870748, 0.011337868480725623, 0.011337868480725623, 0.015873015873015872, 0.015873015873015872, 0.015873015873015872, 0.015873015873015872, 0.013605442176870748, 0.015873015873015872, 0.018140589569160998, 0.011337868480725623, 0.018140589569160998, 0.015873015873015872, 0.011337868480725623, 0.013605442176870748, 0.011337868480725623, 0.011337868480725623, 0.011337868480725623, 0.011337868480725623, 0.009070294784580499, 0.011337868480725623, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.006802721088435374, 0.006802721088435374, 0.006802721088435374, 0.009070294784580499, 0.011337868480725623, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499, 0.009070294784580499]
#
# map_dict = \
# {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 43, 41: 44, 42: 48, 43: 50, 44: 51, 45: 53, 46: 54, 47: 55, 48: 56, 49: 61, 50: 62, 51: 63, 52: 65, 53: 66, 54: 68, 55: 70, 56: 71, 57: 72, 58: 73, 59: 74, 60: 77, 61: 78, 62: 79, 63: 81, 64: 84, 65: 85, 66: 86, 67: 87, 68: 88, 69: 90, 70: 93, 71: 94, 72: 96, 73: 97, 74: 99, 75: 102, 76: 103, 77: 106, 78: 107, 79: 108, 80: 109, 81: 111, 82: 112, 83: 115, 84: 116, 85: 119, 86: 121, 87: 124, 88: 125, 89: 130}
#
# xs = np.array(list(map_dict.values()))
# ys = np.array(our)[xs]
#
# print(len(xs))
# print(len(ys))
#
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.17, bottom=0.15)
# plt.plot(ys)
#
# #plt.yticks(np.linspace(0., 0.15, 4), fontsize=12)
#
# plt.ylabel("Fractional error", fontsize=12)
# plt.xlabel("Number of steps", fontsize=12)
# #plt.xticks(np.linspace(0, 100, 6), fontsize=12)
# plt.tight_layout()
# plt.show()