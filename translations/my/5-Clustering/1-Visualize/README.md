# clustering မိတ်ဆက်ခြင်း

Clustering သည် [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) ၏အမျိုးအစားတစ်ခုဖြစ်ပြီး ဒေတာစုစည်းမှုသည် မှတ်တမ်းမထားသော သို့မဟုတ် ၎င်း၏အင်ပുട്ടများကို ယခင်သတ်မှတ်ထားသော ထုတ်လွှင့်ချက်များနှင့် မကိုက်ညီကြောင်း ခန့်မှန်းထားသည်။ ၎င်းသည် မှတ်တမ်းမထားသော ဒေတာကို အမျိုးမျိုးသော 알고리즘များအသုံးပြု၍ စီစဉ်ကာ ဒေတာအတွင်းရှိ နမူနာအတိုင်း အုပ်စုများပေးသည်။

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 အပေါ်ကပုံကိုနှိပ်၍ ဗီဒီယိုကြည့်ပါ။ သင်သည် clustering ဖြင့် machine learning ကိုလေ့လာသောအခါ Nigerian Dance Hall သီချင်းများကိုအားကျပါစေ - ၎င်းသည် ၂၀၁၄ ခုနှစ်တွင် PSquare မှ ထွက်ရှိခဲ့သောအမြင့်ဆုံးသီချင်းတစ်ပုဒ်ဖြစ်သည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### မိတ်ဆက်ခြင်း

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) သည် ဒေတာရှာဖွေမှုအတွက် အထူးအသုံးဝင်သည်။ Nigerian ပရိသတ်များက မျိုးစုံသီချင်းများကို ဘယ်လိုစားသုံးကြသည်ဆိုတာတွင် ရေရှည်ဂရုစိုက်စွာ ရှာဖွေရေးနှင့် နမူနာများကို ရှာဖွေကြရအောင်။

✅ clustering ၏ အသုံးအနှုန်းများအကြောင်း တစ်မိနစ် စဉ်းစားကြည့်ပါ။ အပြင်ပတ်ဝန်းကျင်တွင်၊ အထည်ဆောင်ပစ္စည်းအုပ်တစ်စုရှိပါက မိသားစုဝင်တစ်ဦးချင်းစီ၏အထည်များကို သီးခြားခွဲထုတ်ရာတွင် clustering ဖြစ်ပေါ်သည် 🧦👕👖🩲။ ဒေတာသိပ္ပံတွင်၊ user ၏ နှစ်သက်မှုများကို စိစစ်ရာတွင် သို့မဟုတ် မမှတ်တမ်းထားသော dataset ၏ သတ်မှတ်ချက်များကို ဖေါ်ထုတ်ရာတွင် clustering ဖြစ်ပေါ်သည်။ clustering သည် ရောဂါအုပ်စုတစ်ခုကဲ့သို့ ထိန်းသိမ်းရန် ကူညီပေးသည်။

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 အပေါ်ကပုံကိုနှိပ်၍ ဗီဒီယိုကြည့်ပါ - MIT မှ John Guttag မှ clustering အကြောင်း မိတ်ဆက်သည်

အသက်မွေး၀မ်းကြောင်း ပတ်ဝန်းကျင်၌ clustering ကို သုံး၍ ဈေးကွက် segmentation နှင့် အသက်အိုင်းအုပ်စုများ၊ မည်ကဲ့သို့ ပစ္စည်းများ ဝယ်ယူကြသည်ဆိုသည်ကို သတ္တိထားနိုင်သည်။ နောက်ထပ်အသုံးဝင်မှုမှာ မှားယွင်းမှုများ သို့မဟုတ် ခရက်ဒစ်ကဒ် လွှဲပြောင်းမှုများမှ တရားမဝင် လှုပ်ရှားမှုများကို ရှာဖွေရေးဖြစ်နိုင်သည်။ သို့မဟုတ် ဆေးဘက်ဆိုင်ရာ စကင်များမှ ကင်ဆာတောက်များကို ဖော်ထုတ်ရန် clustering ကိုသုံးနိုင်သည်။

✅ ဘဏ်လုပ်ငန်း၊ အွန်လိုင်းစျေးဝယ်ခြင်း သို့မဟုတ် စီးပွားရေး ပတ်ဝန်းကျင်တွင် clustering ကို တွေ့ကြုံခံစားဖူးသည်ကို တစ်မိနစ်စဉ်းစားကြည့်ပါ။

> 🎓 စိတ်ဝင်စားဖွယ်ကောင်းသည်မှာ cluster သုံးသပ်ချက်ဟာ ၁၉၃၀ ပြည့်နှစ်များက လူမှုမူနှင့် စိတ်ပညာနယ်ပယ်များမှလာသည်။ ၎င်းကို ဘယ်လိုအသုံးပြုခဲ့တာလဲဟု စဉ်းစားနိုင်ပါသလား?

အခြားပုံစံတွင် ရှာဖွေမှု ရလဒ်များကို အုပ်စုဖွဲ့နိုင်သည် - ဥပမာ စျေးဝယ်လင့်ခ်များ၊ ပုံများ သို့မဟုတ် သုံးသပ်ချက်များဖြင့်။ clustering သည် အရမ်းကြီးမားသော ဒေတာစုစည်းမှုရှိပြီး ဆက်လက် စိစစ်တွက်ချက်ရန် အကြမ်းဖျင်းနည်းလမ်းဖြင့် အသုံးပြုလိုသောအခါ အသုံးကျသည်။ ထို့ကြောင့် မတူညီသည့် မော်ဒယ်များတည်ဆောက်ခြင်းမတိုင်မီ ဒေတာကို သင်ယူနိုင်ရန် အသုံးပြုသည်။

✅ သင်၏ဒေတာအား အုပ်စုများအနေဖြင့် စုစည်းပြီးနောက်၊ အဲဒီအုပ်စုကို cluster Id ဖြင့် သတ်မှတ်၍ ဒီနည်းလမ်းက ဒေတာစုစည်းမှု၏ ကိုယ်ရေးကာကွယ်မှုကို ထိန်းသိမ်းရာတွင် အသုံးဝင်နိုင်သည်။ အစားတစ်ခုချင်းစီအား cluster id ဖြင့် ကိုယ်စားပြု၍ သီးခြားသိရှိစေသော ဒေတာများဖြင့် မသိသာစေခြင်းဖြစ်သည်။ cluster Id ကို အခြားသူများအစား သုံးရသော အခြားအကြောင်း ပြောပြနိုင်ပါသလား။

clustering နည်းပညာများကို ပိုမိုနက်ရှိုင်းစွာ လေ့လာရန် [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott) ကို ကြည့်ပါ  
## clustering စတင်ခြင်း

[Scikit-learn သည် clustering လုပ်ရန် နည်းလမ်းများ များစွာ](https://scikit-learn.org/stable/modules/clustering.html) ပေးပို့ပေးသည်။ သင်ရွေးချယ်မည့်အမျိုးအစားသည် သုံးရန်အခြေအနေမှာ မူတည်သည်။ စာရွက်စာတမ်းအရ၊ နည်းလမ်းတိုင်းတွင် အကြောင်းအရင်းများ ရှိသည်။ ရိုးရှင်းစွာ အောက်တွင် Scikit-learn သတင်းအချက်အလက်တွင် ပါဝင်သော နည်းလမ်းများနှင့် သုံးစေရေး ဖြစ်စဉ်များကို တပ်ဆင်ထားသည် -

| နည်းလမ်းအမည်                | အသုံးပြုမှု                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | ပေါ်ပေါက်နေသော ရည်ရွယ်ချက်များ၊ inductive                                             |
| Affinity propagation         | cluster များအနေ အများ၊ မညီမျှမှုရှိသည်၊ inductive                                       |
| Mean-shift                   | cluster များအနေ အများ၊ မညီမျှမှုရှိသည်၊ inductive                                       |
| Spectral clustering          | cluster အနည်းငယ်၊ ထပ်တူညီမျှ cluster များရှိသည်၊ transductive                                       |
| Ward hierarchical clustering | cluster များအနေ အများ၊ ကန့်သတ်ထားသော cluster များ, transductive                               |
| Agglomerative clustering     | cluster များအနေ အများ၊ ကန့်သတ်ထားသော၊  ဥယျာဉ်ဥပမာ Euclidean မဟုတ်သော အကွာအဝေးများ၊ transductive               |
| DBSCAN                       | non-flat geometry, မညီမျှသော cluster များ, transductive                       |
| OPTICS                       | non-flat geometry, မညီမျှပြီး အနူးအနန္တ စွမ်းအင်ရှိသော cluster များ, transductive |
| Gaussian mixtures            | flat geometry, inductive                                               |
| BIRCH                        | အကြီးစား dataset အသုံးပြုခြင်းနှင့် ထွက်ရှိမှုများပါရှိသည်, inductive                                |

> 🎓 cluster များကို ဘယ်လိုဖန်တီးသလဲဆိုတာသည် ဒေတာအမှတ်များကို အုပ်စုအဖြစ် စုပေါင်းခြင်း နည်းလမ်းနှင့် အတူ များစွာဆိုင်သည်။ အချို့အကြောင်းအရာကို ခွဲခြမ်းစိတ်ဖြာကြရအောင် -
>
> 🎓 ['Transductive' နှင့် 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transductive inference သည် ကြည့်ရှုခံရသော သင်ကြားမှုကိစ္စများမှ သတ္တိပြုသည့် စမ်းသပ်မှုကိစ္စများကို တိုက်ရိုက်ချိတ်ဆက်သည်။ Inductive inference သည် သင်ကြားမှုကိစ္စများမှ အထွေထွေစည်းမျဉ်းများသို့ ပြောင်းလဲသည့်နောက် သတ်မှတ်မှုများအား စမ်းသပ်မှုကိစ္စများတွင် အသုံးပြုသည်။
> 
> နမူနာ - သင်တွင် သင်ခန်းစာတစ်ခုတွင် ဒီတာစုတစ်ခုမှာ ဒေတာအမှတ်တစ်စိတ်တစ်ပိုင်းသာ မှတ်တမ်းထားသည်ဟု စဉ်းစားပါ။ အချို့ဟာ 'record' များဖြစ်ပြီး အချို့ 'cd' များဖြစ်သည်၊ အချို့လည်း ရှင်းလင်းချက်မရှိပါ။ သင်တစ်ဦးအနေဖြင့် ရှင်းလင်းချက်မရှိသော အချို့ရှိရာတွင် label များပေးရန်တာ၀န်ရှိသည်။ inductive နည်းလမ်းအသုံးပြုပါက 'record' နှင့် 'cd' များ ရှာဖွေရန် ကိုယ်တတ်နိုင်သည့် model တစ်ခုဘယ်လိုတည်ဆောက်မည်ဆိုသည့် မော်ဒယ်ကို သင်ကြားပြီး သီးခြားတော့ေတာမဲ့ဒေတာတွင် သတ်မှတ်ချက်များကို အသုံးပြုသွားမည်ဖြစ်သည်။ ၎င်းနည်းလမ်းသည် ထိုအမှတ်အသားမရှိသော 'cassette' များကို သတ်မှတ်ရာတွင် အခက်အခဲရှိမည်ဖြစ်သည်။ Transductive နည်းလမ်း ဒါမှမဟုတ် သိသာသည့် ဒေတာများကို အုပ်စုသို့ ပုံစံပေါ်တင်ပြီး သတ်မှတ်ချက်များကို ဆက်လက် မျှဝေသည်။ ဒီကိစ္စတွင် cluster များသည် 'ကမ္ဘာပတ်သံ္ဂါး' များနှင့် 'ချောင်းစတုရန်းအမျိုးအစားသံဂါး' များကို ဖော်ပြနိုင်သည်။
> 
> 🎓 ['Non-flat' နှင့် 'flat' geometry](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> သင်္ချာဆိုင်ရာအသုံးအနှုန်းမှ ရရှိသော non-flat နှင့် flat geometry သည် အချက်အလက်များအကြား အကွာအဝေးကို 'flat' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) သို့မဟုတ် 'non-flat' (non-Euclidean) ခြေလမ်းဖြင့် တိုင်းတာသည်။
>
> 'Flat' သည် Euclidean Geometry ကို ရည်ညွှန်းပြီး non-flat သည် non-Euclidean Geometry ကို ရည်ညွှန်းသည်။ Machine learning နှင့် သင်္ချာဆိုင်ရာ နယ်ပယ်များသည် ဒေတာအမှတ်များအကြား အကွာအဝေး တိုင်းတာခြင်းအတွက် စည်းကမ်းတစ်ခုရှိသည်။ စည်းကမ်းသည် ဒေတာအမျိုးအစားပေါ်မူတည်၍ 'flat' သို့မဟုတ် 'non-flat' ဖြစ်နိုင်သည်။ [Euclidean အကွာအဝေးများ](https://wikipedia.org/wiki/Euclidean_distance) သည် အချက်နှစ်ချက်အကြား မှတ်သားထားသော ရှည်လျားသော အိုင်လိုင်းအတိုင်း တိုင်းတာသည်။ [Non-Euclidean အကွာအဝေးများ](https://wikipedia.org/wiki/Non-Euclidean_geometry) သည် ဖိုက်လာများအတိုင်း တိုင်းတာသည်။ သင်၏ဒေတာကို ကိုးကား၍ ကြည့်လျှင် မတည်မှန်ဘဲ ရပ်တည်နေသည်ဟု တွေးပါက သီးခြားသော 알고리즘 အသုံးပြုရမည် ဖြစ်သည်။
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/my/flat-nonflat.d1c8c6e2a96110c1.webp)
> ပုံရိပ်ကို [Dasani Madipalli](https://twitter.com/dasani_decoded) ဖန်တီးသည်
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> cluster များသည် သတ်မှတ်အချက်တစ်ခု၏ အကွာအဝေး matrix ဖြင့် သတ်မှတ်သည်၊ ဥပမာ အချက်များအကြား အကွာအဝေး ဖြစ်သည်။ ဒီအကွာအဝေးကို မတည့်သော နည်းလမ်းအနည်းငယ်ဖြင့် တိုင်းတာနိုင်သည်။ Euclidean cluster များသည် ဖော်ပြသည့်အချက်များ၏ ပျမ်းမျှ တန်ဖိုးဖြင့် သတ်မှတ်သည်။ ဤ cluster များတွင် 'centroid' သို့မဟုတ် ဗဟို အချက်ရှိသည်။ အကွာအဝေးကို အဆိုပါ centroid သို့ မျက်မှောက်တင့် မထားသည်။ Non-Euclidean အကွာအဝေးမှာ 'clustroids' ဖြစ်၍ အခြားအချက်များနှင့်ရင်းနှီးနီးနီးဆုံး အချက်ကို ဆိုလိုသည်။ Clustroids ကို အမျိုးမျိုးသတ်မှတ်နိုင်သည်။
> 
> 🎓 ['Constrained'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) သည် unsupervised နည်းလမ်းတွင် 'semi-supervised' သင်ယူမှုကို ဖြည့်စွက်သည်။ အချက်အလက်များအကြား ဆက်နွယ်မှုများကို 'cannot link' သို့မဟုတ် 'must link' အဖြစ် ပြသခြင်းဖြင့် စည်းကမ်းချထားသည်။
>
> ဥပမာ - အာလဂိုရီသမ့်သည် မှတ်တမ်းမထားသော ဒေတာတစ်စု သို့မဟုတ် အချို့မှတ်တမ်းထားသော ဒေတာတွင် လွတ်လပ်စွာ လည်ပတ်ပါက ဖန်တီးသည့် cluster များ သက်ရောက်မှုကျသော မဟုတ်သော ပမာဏဖြစ်နိုင်သည်။ ဥပမာအရ cluster များသည် 'ကမ္ဘာပတ်သံဂါးများ', 'စတုရန်းသံဂါးများ', 'သဘောနည်းသုံးပုံပုံ', 'ခေါက်ဆွဲများ' ဟူ၍ ခွဲခြားထားနိုင်သည်။ စည်းကမ်းတချို့ ("ပစ္စည်းသည် ပလပ်စတစ်မှ ဖြစ်ရမည်", "ပစ္စည်းသည် သံဂါးဖွဲ့နိုင်ရမည်") နှင့်အတူ algorithm သိမ်းဆည်းမှုကို ပိုမိုကောင်းမွန်သော ရွေးချယ်မှုများ ပြုလုပ်ဖို့ ကူညီပေးသည်။ 
> 
> 🎓 'Density'
> 
> "အသံညှိမှား" ဖြစ်သော ဒေတာကို "dense" ဟု သတ်မှတ်သည်။ ၎င်း cluster တစ်ခုလျှင် အချက်များအကြား အကွာအဝေးသည် တော့ ဒေတာသည် ပိုမို သို့မဟုတ် နည်းပါးသည့် density ရှိသည်။ ဒါကြောင့် တိုင်းတာပြီး သင့်တော်သော clustering နည်းစနစ်ဖြင့် စိစစ်ရမည်။ [ဤဆောင်းပါး](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) သည် noisy ဒေတာတွင် cluster density မတည့်မှုများကို ရှာဖွေရေးအတွက် K-Means clustering နှင့် HDBSCAN အသုံးပြုပြီး ကွာခြားချက်ကို ဖော်ပြသည်။

## clustering 알고리즘 များ

clustering 알고리즘 များ ၁၀၀ ကျော်ရှိသည်၊ သုံးစွဲမှုသည် ဖြစ်ပေါ်သော ဒေတာ အမျိုးအစားပေါ် မှီတည်သည်။ အကြီးစား အချို့ကို ဆွေးနွေးကြပါစို့ -

- **Hierarchical clustering**။ အရာဝတ္ထုတစ်ခုကို အနီးရှိ အရာဝတ္ထုတစ်ခု၏ အကွာအဝေးအပေါ် မူတည်၍ အမျိုးအစားခွဲခြားခြင်း ဖြစ်သည်၊ cluster များသည် အဖွဲ့ဝင်များ၏ အကွာအဝေး ပြန်လည်ချိတ်ဆက်မှုတစ်ခုဖြစ်လာသည်။ Scikit-learn ၏ agglomerative clustering သည် hierarchical ဖြစ်သည်။

   ![Hierarchical clustering Infographic](../../../../translated_images/my/hierarchical.bf59403aa43c8c47.webp)
   > ပုံရိပ်ကို [Dasani Madipalli](https://twitter.com/dasani_decoded) ဖန်တီးသည်

- **Centroid clustering**။ ဤကျော်ကြားသော 알고리즘 တွင် 'k' ဟူသော cluster အရေအတွက်ကိုရွေးချယ်ပြီးနောက်၊ algorithm သည် cluster ၏ ဗဟိုအချက်ကို ရှာဖွေပြီး အချက်များကို ဤနေရာအနီးတွင် စုစည်းသည်။ [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) သည် centroid clustering ၏ ကြားထဲဖြစ်သည်။ ဗဟိုသည် အနီးဆုံး ပျမ်းမျှတန်ဖိုးမှ သတ်မှတ်သည်။ cluster မှ ကွာရှင်းချက်ကို လျော့နည်းစေသည်။

   ![Centroid clustering Infographic](../../../../translated_images/my/centroid.097fde836cf6c918.webp)
   > ပုံရိပ်ကို [Dasani Madipalli](https://twitter.com/dasani_decoded) ဖန်တီးသည်

- **Distribution-based clustering**။ စာရင်းဇယားဖြစ်စဉ်များအပေါ် အခြေခံပြီး၊ ဒေတာအချက်တစ်ခုcluster တွင် ပိုင်ဆိုင်မှုဖြစ်နိုင်ခြေကို ဖော်ထုတ်ရန် ဦးတည်သည်။ Gaussian mixture နည်းလမ်းများသည် ဤအမျိုးအစားတွင် ပါဝင်သည်။

- **Density-based clustering**။ ဒေတာအချက်များအား အထက်ပါ cluster ၏ စည်းကမ်းအပေါ် အခြေခံ၍ မြုပ်ကွေးခြင်း၊ သို့မဟုတ် တစ်ဦးနောက်တစ်ဦး စုပေါင်းမှုအပေါ် အခြေခံ၍ ခွဲခြားသည်။ အုပ်စုမှ ဝေးကွာသောအချက်များကို ထွက်ခွာသူများ သို့မဟုတ် ဆူညံ့သံများအဖြစ် သတ်မှတ်သည်။ DBSCAN, Mean-shift နှင့် OPTICS သည် ဤအမျိုးအစားမှာ ပါဝင်သည်။

- **Grid-based clustering**။ မျိုးစုံတန်းစားဒေတာများအတွက် grid ဖန်တီးပြီး ဒေတာအား grid ၏ ဆဲလ်များအလိုက် ခွဲခြားခြင်းဖြင့် cluster များတည်ဆောက်သည်။

## လေ့ကျင့်ခန်း - သင်၏ ဒေတာကို cluster ဖွဲ့ပါ

cluster ဖွဲ့ခြင်းသည် ထိရောက်သော နည်းလမ်းဖြစ်ပြီး သင့်ဒေတာကို ထိရောက်စွာ မြင်သာစေသော ခြုံကြည့်မှုဖြင့် ကူညီပါသည်။ ဒီလေ့ကျင့်ခန်းက သင့် ဒေတာအမျိုးအစားအတွက် အသုံးဝင်သော clustering နည်းလမ်းကို ရွေးချယ်ခြင်းကို ကူညီပါလိမ့်မည်။

၁။ ယခု ဖိုင်လ်ဒီမှာရှိသော [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) ဖိုင်ကို ဖွင့်ပါ။

၁။ သင့်တင့်သော ဒေတာမြင်သာရေးအတွက် `Seaborn` package ကို import လုပ်ပါ။

    ```python
    !pip install seaborn
    ```

၁။ [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) မှ သီချင်းဒေတာကို ထည့်သွင်းပါ။ သီချင်းပေါ် ဆိုင်ရာဒေတာနှင့် dataframe တစ်ခု ဖန်တီးပါ။ စတင်စမ်းသပ်ရန် အသုံးပြုရန် library များ import ပြုလုပ်ပြီး ဒေတာကို ပိတ်မပါနှင့် -

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    ဒေတာ အစောပိုင်းလိုင်း အနည်းငယ်ကို စစ်ဆေးပါ -

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
    | 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
    | 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
    | 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. dataframe အကြောင်းအရာတွေ ရယူရန် `info()` ကိုခေါ်ဆောင်ပါ။

    ```python
    df.info()
    ```

   အောက်ပါနမူနာအတိုင်း output ကိုမြင်ရပါမယ်။

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. `isnull()` ကိုခေါ်ပြီး null တန်ဖိုးတွေ ရှိမရှိ နှစ်ကြိမ်စစ်ဆေးပါ၊ စုစုပေါင်း 0 ဖြစ်ဖို့ လိုသည်။

    ```python
    df.isnull().sum()
    ```

    အဆင်ပြေပါတယ်။

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. ဒေတာအကြောင်းအရာကို ဖော်ပြပါ။

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 cluster ပြုလုပ်ရာတွင် အမှတ်အသားမလိုအပ်တဲ့ unsupervised method တစ်ခုဖြစ်လို့ ဒီ data ကို label တွေနဲ့ပြသတာဘာကြောင့်လဲ? ဒေတာဇာတိစစ်ဆေးရာတွင် အချက်အလက်အတွက် အဆင်ပြေသော်လည်း clustering algorithm များအလုပ်လုပ်ရန်အတွက် label မလိုအပ်ပါ။ ကော်လံခေါင်းစဉ်တွေဖယ်ရှားပြီး ကော်လံနံပါတ်အတိုင်းအချက်အလက်ကို ရည်ညွှန်းနိုင်ပါသည်။

ဒေတာရဲ့ပုံမှန်တန်ဖိုးတွေကို ရှုပါ။ popularity က '0' ဖြစ်နိုင်သည်၊ ဒါက ဂီတသီချင်းတွေရဲ့ အဆင့်သတ်မှတ်ချက်မရှိကြောင်းကို ဖော်ပြသည်။ ထိုကွောငျ့ အနည်းငယ်ဖယ်ရှားပါမယ်။

1. လူကြိုက်ဆုံးအမျိုးအစားများကို အတန်းလိုက်အကြည့် ရှာဖွေရန် barplot ကိုအသုံးပြုပါ။

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/my/popular.9c48d84b3386705f.webp)

✅ ထိပ်ဆုံးတန်ဖိုးများ ပိုများစွာကြည့်လိုပါက top `[:5]` ကို ပိုကြီးသောတန်ဖိုးသို့ပြောင်းပါ၊ ဒါမှမဟုတ်အားလုံးကြည့်လိုလျှင် ဖယ်ရှားနိုင်သည်။

မှတ်ချက်။ အထက်ဆုံး genre ကို 'Missing' ဟုဖော်ပြပါက Spotify မှ အမျိုးအစားအလိုက်ခွဲခြားမှု မရှိသောကြောင့်၊ ထိုအပိုင်းကို ဖယ်ရှားကြပါ။

1. missing data ကို ဖယ်ရှားရန် စစ်ထုတ်ပါ။

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ယခု genre များကို ပြန်စစ်ပါ။

    ![most popular](../../../../translated_images/my/all-genres.1d56ef06cefbfcd6.webp)

1. ထို dataset မှာ အထက်ဆုံး သုံးလွှာရှိ genre များမှာ အလွန်ထင်ရှားသည်။ `afro dancehall`၊ `afropop` နှင့် `nigerian pop` တွင်အာရုံစိုက်ပြီး၊ popularity သည် 0 ဖြစ်သော data များကို ဖယ်ရှားပါ (dataset မှာ popularity ခွဲခြားချက် မရှိတဲ့ အချက်အလက် ဖြစ်ပြီး၊ ငယ်နားလည်မှုအတွက် သံသယတစေပါသည်)။

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. ဒေတာတွင် အားကောင်းသောဆက်စပ်မှု ရှိမရှိ အမြန်စမ်းသပ်ပါ။

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/my/correlation.a9356bb798f5eea5.webp)

    တစ်ခုတည်းသောအားကောင်းသောဆက်စပ်မှုမှာ `energy` နှင့် `loudness` ကြား ဖြစ်သည်၊ ဂီတအသံရည်ကြောင်း အချို့အတွက် သာမက လည်း စိတ်မရှုပ်ပါ။ အခြား ဆက်စပ်မှုများမှာ ချောင့်ချင်းနည်းပါးသည်။ clustering algorithm သည် ဒီဒေတာကနေ ဘယ်လိုအကြောင်းအရာ ဆွဲယူနိုင်မလဲ စိတ်ဝင်စားစရာ ဖြစ်ပါသည်။

    > 🎓 ဆက်စပ်မှုဟာ အသက်မွေးဝမ်းကြောင်းကို သက်သေပြခြင်းမဟုတ်ပါ! ကျွန်ုပ်တို့ဆက်စပ်မှုကိုသာ သက်သေပြထားပြီး causation ကိုနှင့် မသက်သေပြပါ။ [amusing web site](https://tylervigen.com/spurious-correlations) တွင်၎င်းအကြောင်း ဗေဒင်များပြပါသည်။

ဒီ dataset မှာ သီချင်းရဲ ့ လူကြိုက်မှုနှင့် danceability သဘောဆန်းမှု တည့်တည့်သေသေ ရှိမရှိ ရှာကြည့်ပါ။ FacetGrid မှာ အမျိုးအစားမရွေးဟုဆိုပေမယ့် စက်ဝိုင်းပုံစံများ ရှိသူကို တွေ့နိုင်သည်။ Nigerian လူငယ်များသည် ထိုအမျိုးအစားအတွက် တစ်ခုတည်းသော danceability အဆင့်တွင် အတူညီနေသည်လား?

✅ မတူညီသော ဒေတာချက်များ (energy, loudness, speechiness) နှင့် အမျိုးအစား အမျိုးမျိုး သို့မဟုတ် မတူညီသော ဂီတ styl ဂျန်း များကို စမ်းသပ်ကြည့်ပါ။ ဘာတွေ တွေ့ရှိနိုင်သလဲ စူးစမ်းကြည့်ပါ။ `df.describe()` ဇယားက ဒေတာစု အခြေခံဖြန့်ချိမှုကို ဖော်ပြထားသည်။

### အလုပ်လေ့ကျင့်ခန်း - ဒေတာဖြန့်ချိမှု

အထက်ပါ သုံးခုအမျိုးအစားများသည် သူတို့၏ popularity အပေါ် မူတည်၍ danceability အား တွေးကြည့်ချက်များမှာ ထူးခြားရွေးချယ်မှုရှိသလား?

1. အထက်ဆုံး သုံးမျိုး အမျိုးအစားရှိ popularity နှင့် danceability ကို x နှင့် y axis ပေါ်တွင် စစ်ဆေးပါ။

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    လူကြိုက်မှုနှင့် danceability တွင် စက်ဝိုင်းပုံစံအားဖြင့် data အနီးအနား ကျယ်ပြန့်နေမှုကို တွေ့ရမည်။

    > 🎓 ဤနမူနာတွင် KDE (Kernel Density Estimate) ပုံစံကို အသုံးပြုထားပြီး continuous probability density curve ဖြင့် data ကို ကိုယ်စားပြုသည်။ ၎င်းသည် မျိုးစုံဖြန့်ချိမှု များကိုလည်း သဘာဝနှင့်လိုက်ဖက်စွာ ဖော်ပြနိုင်ပါသည်။

    စုစုပေါင်းနှင့် အထက်ပါသုံးမျိုးကြား popularity နှင့် danceability များသည် အနည်းငယ်သေးငယ်ပြီး ချိန်ညှိမှုရှိသည်ဆိုသော သဘော ရှိသည်။ ဒီလို ချိန်ညှိမှုရှိတဲ့ဒေတာတွေကို cluster ခွဲရာတွင် စိန်ခေါ်မှု ဖြစ်ပါလိမ့်မယ်။

    ![distribution](../../../../translated_images/my/distribution.9be11df42356ca95.webp)

1. scatter plot တစ်ခု ဖန်တီးပါ။

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    တူညီတဲ့ axis များဖြင့် scatterplot ပြသသည့် ပုံစံဟာ တူညီသော ချိန်ညှိမှုပုံစံတစ်ခု လိုက်နာရှိသည်။

    ![Facetgrid](../../../../translated_images/my/facetgrid.9b2e65ce707eba1f.webp)

စုစုပေါင်း cluster ခွဲရာတွင် scatterplot များကို data cluster များကို မြင်မြင်သာသာ ဖော်ပြရန် အသုံးပြုနိုင်ပြီး ဤ visualization ပုံစံကို ကျွမ်းကျင်ခြင်းမှာ အထောက်အကူ ဖြစ်ပါသည်။ နောက်ခန်းမှာ ခွဲထုတ်ထားသော ဒေတာကိုအသုံးပြုပြီး k-means clustering ဖြင့် ဒေတာအုပ်စုများကို ရှာဖွေနိုင်မည်။

---

## 🚀စိန်ခေါ်မှု

နောက်ခန်းတွင် ပြုလုပ်မည့် ဇယားများအတွက် အမျိုးအစား clustering algorithm များ ရှာဖွေပြီး ထုတ်လုပ်မှုပတ်ဝန်းကျင်တွင် အသုံးပြုနိုင်သော အခြေအနေများ ရေးဆွဲပါ။ Clustering များသည် မည်သည့်ပြဿနာများကို ဖြေရှင်းရန် ကြိုးပမ်းထားပါသနည်း?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာမှု

Clustering algorithm များကို အသုံးပြုခြင်းအားမပြုမီ၊ ဒေတာစု၏ သဘာဝကို နားလည်သည်မှာ အရေးကြီးပါသည်။ ဤအကြောင်းကို [ဤနေရာတွင်](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html) ဖတ်ရှုနိုင်ပါသည်။

[ဒီအရေးအသား](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) မှ clustering algorithm များ၏ ကွဲပြားမှုများကို ဒေတာပုံစံအမျိုးမျိုးအပေါ် အခြေခံပြီး ရှင်းပြထားသည်။

## အပ်တိုးမိတ်

[Clustering အတွက် အခြား visualizations များကို သုတေသန လုပ်ရန်](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ပြောကြားချက်**
ဤစာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးပမ်းနေသော်လည်း၊ စက်ကိရိယာဘာသာပြန်ခြင်းများတွင် အမှားများ သို့မဟုတ် မှားယွင်းချက်များ ပါဝင်နိုင်ကြောင်း သတိပြုပါရန် လိုအပ်ပါသည်။ မူလစာတမ်းကို မူရင်းဘာသာဖြင့်သာ ယုံကြည်စိတ်ချရသော အချက်အလက်အဖြစ် သတ်မှတ်သင့်သည်။ အရေးကြီးသည့် သတင်းအချက်အလက်များအတွက် ပရော်ဖက်ရှင်နယ် လူသားဘာသာပြန်သူဝန်ဆောင်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်ချက်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုကွာခြားမှုများ သို့မဟုတ် မမှန်ကန်သော အသုံးပြုမှုများအတွက် ကျွန်ုပ်တို့ တာဝန်မခံပါ။
<!-- CO-OP TRANSLATOR DISCLAIMER END -->