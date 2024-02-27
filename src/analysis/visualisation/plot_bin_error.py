import matplotlib.pyplot as plt
import numpy as np

our_mean = [ 0.13850415512465375,0.1920590951061865,0.30332409972299174,0.12049861495844875,0.11495844875346262,0.12880886426592797,0.14312096029547555,0.11726685133887349,0.09141274238227147,0.08264081255771007,0.07063711911357341,0.06602031394275161,0.06463527239150509,0.06140350877192983,0.055401662049861494,0.05355493998153279,0.05170821791320407,0.04755309325946445,0.0443213296398892,0.04062788550323176,0.03831948291782086,0.036934441366574325,0.03647276084949214,0.03462603878116343,0.03462603878116343,0.035087719298245605,0.03324099722991689,0.03139427516158818,0.029547553093259463,0.02723915050784857,0.02770083102493075,0.026777469990766394,0.026777469990766394,0.025854108956602034,0.021237303785780242,0.0221606648199446,0.0221606648199446,0.024469067405355496,0.023084025854108958,0.021237303785780242,0.02077562326869806,0.02077562326869806,0.021237303785780242,0.020313942751615882,0.018928901200369344,0.018928901200369344,0.018928901200369344,0.018467220683287166,0.018005540166204984,0.018467220683287166,0.017543859649122806,0.018928901200369344,0.018928901200369344, ]
our_std = [ 0.0,0.042000241608658014,0.14990688897655846,0.013163968078839572,0.02108116353985388,0.0183572440924017,0.03322817029630025,0.02364958167103545,0.019846893130382893,0.02258462568538321,0.019175032001195923,0.015291315938761952,0.013155869666458916,0.009245145149815689,0.010364701902420888,0.011824790835517724,0.009684292226871206,0.0060724591126343035,0.007996541124510053,0.006529148487410413,0.005858069039912059,0.007092470681319122,0.008062903599525845,0.008723196504012551,0.008723196504012551,0.008412219371324374,0.012284519571237367,0.012353728679833472,0.012761103380503465,0.01124265527413289,0.010964304789508696,0.01245682138802589,0.013058296974820824,0.011606468227125149,0.009551320805898983,0.009325489324434052,0.00985879801664941,0.006674437809234055,0.006909801268280594,0.004708235931295277,0.005243682683102746,0.0047308175281438585,0.004708235931295277,0.004129396080332023,0.0051617451004150276,0.0051617451004150276,0.004907731215482295,0.004972451345461223,0.004730817528143859,0.004428283955044063,0.004972451345461223,0.005161745100415027,0.005161745100415027, ]

base_mean = [ 0.13573407202216067,0.20360110803324097,0.24192059095106186,0.24192059095106186,0.2031394275161588,0.17174515235457063,0.15512465373961218,0.16204986149584485,0.12788550323176362,0.12742382271468145,0.11680517082179132,0.08402585410895662,0.08679593721144968,0.07710064635272391,0.06694367497691597,0.08587257617728533,0.08079409048938135,0.07987072945521699,0.07063711911357341,0.06602031394275161,0.055401662049861494,0.07017543859649122,0.08125577100646351,0.07479224376731301,0.057248384118190214,0.06694367497691597,0.05909510618651894,0.05170821791320407,0.060018467220683276,0.04939981532779317,0.0507848568790397,0.04755309325946445,0.04847645429362881,0.042474607571560484,0.04201292705447831,0.038781163434903045,0.036011080332409975,0.03785780240073869,0.03601108033240997,0.03831948291782087,0.035549399815327794,0.04062788550323176,0.042012927054478295,0.03647276084949216,0.03785780240073869,0.042012927054478295,0.03831948291782086,0.03554939981532779,0.033702677746999074,0.03277931671283472,0.031855955678670354, ]
base_std = [ 0.0,0.02631578947368421,0.05987624316481431,0.04584373624120558,0.04584373624120557,0.045093685861772044,0.0557468470844823,0.023217527166537685,0.015946367533860147,0.022333123956505677,0.03147902850181124,0.03585685658162069,0.0359636973114505,0.024486483113898647,0.02025615047874451,0.015751359288302846,0.014867259638139074,0.02417108054055111,0.017791180856876913,0.01425239985266217,0.018990732965099847,0.01799962021202025,0.013348875618468108,0.009728212144831706,0.0072705520535658475,0.014606917838925555,0.00633024432169995,0.006909801268280594,0.019500195828202106,0.006674437809234052,0.01044663757985666,0.005858069039912058,0.008271686457603379,0.008562897964446632,0.007226443140581038,0.010608610612258594,0.009047053528285052,0.021733337573296065,0.012592965555850282,0.01124265527413289,0.009517787686098017,0.004972451345461222,0.006674437809234052,0.008374126106748408,0.007614230148878413,0.01089605145079035,0.004051230095748901,0.006072459112634304,0.012223640161471585,0.006674437809234053,0.006545451005890039, ]

svc_mean = [ 0.13850415512465375,0.3638042474607572,0.2206832871652816,0.2031394275161588,0.172668513388735,0.13481071098799632,0.12557710064635272,0.07479224376731303,0.08587257617728532,0.13204062788550325,0.0817174515235457,0.0881809787626962,0.10941828254847646,0.08079409048938135,0.0974145891043398,0.08125577100646353,0.06971375807940905,0.061865189289011996,0.0530932594644506,0.06001846722068329,0.06463527239150509,0.05355493998153279,0.03647276084949214,0.056786703601108025,0.04801477377654662,0.04709141274238227,0.05263157894736842,0.04847645429362881,0.05216989843028624,0.04155124653739612,0.04524469067405356,0.042012927054478295,0.03970452446906741,0.04155124653739612,0.04339796860572484,0.042474607571560484,0.04939981532779317,0.04662973222530009,0.04847645429362881,0.05771006463527239,0.0530932594644506,0.05124653739612189,0.05124653739612187,0.05493998153277932,0.060480147737765465,0.04986149584487534,0.049399815327793174,0.05355493998153279,0.05909510618651894,0.05447830101569714,0.04986149584487535,0.04662973222530009, ]
svc_std = [ 0.0,0.0026116593949641657,0.0026116593949641657,0.049621528504319126,0.014364126672302911,0.016975786067267077,0.01828161576474915,0.01566995636978499,0.014124707793885829,0.039779610763689785,0.010083254694294346,0.008964214145405174,0.037804277185474196,0.012431128363653055,0.048730810513116465,0.02698363655808395,0.0216596567049675,0.019630924861929175,0.008062903599525847,0.026745611002696654,0.026311739332917836,0.01531221048178855,0.008964214145405169,0.015235457063711912,0.015724271806026223,0.013380772618826814,0.0076700128004783714,0.010083254694294346,0.012118563941280414,0.006397232899608043,0.017713135820176248,0.010778040193840028,0.008856567910088124,0.010728485723566251,0.006909801268280593,0.009684292226871205,0.006072459112634305,0.011467906138849696,0.012048004017266748,0.01901877185969158,0.019085897145516516,0.011166561978252838,0.013066455861574247,0.009382456801893618,0.01253358445106023,0.009461635056287717,0.00896421414540517,0.012559067875101979,0.01214491822526861,0.012660488643399893,0.008310249307479225,0.009912701086603824, ]

accs = [our_mean, base_mean, svc_mean]
stds = [our_std, base_std, svc_std]
label = ["Ours", "GPR", "SVC"]

plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.17, bottom=0.15)
for i, (acc, std, l) in enumerate(zip(accs, stds, label)):
    plt.plot(acc, label=l)

    upper = [a + e for a, e in zip(acc, std)]
    lower = [a - e for a, e in zip(acc, std)]
    plt.fill_between(range(len(lower)), lower, upper, alpha=0.3)


plt.ylim([0, 0.2])
plt.yticks(np.linspace(0., 0.2, 5), fontsize=12)
plt.xlim([0, 50])
plt.xticks(np.linspace(0, 50, 6), fontsize=12)

plt.ylabel("Fractional error", fontsize=12)
plt.xlabel("Number of steps", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("../../../images/bin_error.pdf")
plt.show()