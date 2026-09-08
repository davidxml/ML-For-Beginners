# ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ಗೆ ಪರಿಚಯ

ಕಾರ್ಯಕ್ರಮದ ಈ ಭಾಗದಲ್ಲಿ, ನೀವು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ನ ಮೌಲಿಕ ತತ್ವಗಳಿಗೆ ಪರಿಚಯವಾಗುತ್ತೀರಿ, ಇದರಲ್ಲಿ ಇದು 무엇인지, ಅದರ ಇತಿಹಾಸ, ಮತ್ತು ಸಂಶೋಧಕರು ಇದನ್ನು ವಾಸ್ತವಿಕ ಪರಿಸರಗಳಲ್ಲಿ ಅನ್ವಯಿಸುವ ತಂತ್ರಗಳು شاملವಿದೆ. ನಾವು ಈ ರೋಚಕ ML ಲೋಕವನ್ನು گڏಾಗಿ ಅನ್ವೇಷಣೆಯೋಡನೆ ನೋಡೋಣ!

![globe](../../../translated_images/kn/globe.59f26379ceb40428.webp)
> ಚಿತ್ರ <a href="https://unsplash.com/@bill_oxford?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ಬಿಲ್ ಆಕ್ಸ್ಫರ್ಡ್</a> ಅವರಿಂದ <a href="https://unsplash.com/s/photos/globe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ಅನ್ಸ್‌ಪ್ಲ್ಯಾಶ್</a> ನಲ್ಲಿ
   
### ಪಾಠಗಳು

1. [ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಪರಿಚಯ](1-intro-to-ML/README.md)
1. [ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಮತ್ತು ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯ ಇತಿಹಾಸ](2-history-of-ML/README.md)
1. [ನ್ಯಾಯತೆ ಮತ್ತು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್](3-fairness/README.md)
1. [ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ತಂತ್ರಗಳು](4-techniques-of-ML/README.md)
### ಕ್ರೆಡಿಟ್‌ಗಳು

"ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ಗೆ ಪರಿಚಯ" ಒಲುಮೆಯೊಂದಿಗೆ [ಮುಹಮ್ಮದ್ ಸಾಕಿಬ್ ಖಾನ್ ಇನಾನ್](https://twitter.com/Sakibinan), [ಒರ್ನೆಲ್ಲಾ ಅಲ್ಟುನ್ಯಾನ್](https://twitter.com/ornelladotcom) ಮತ್ತು [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಸೇರಿದಂತೆ ತಂಡದ ಸದಸ್ಯರಿಂದ ಬರೆಯಲಾಗಿದೆ

"ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಇತಿಹಾಸ" ಒಲುಮೆಯೊಂದಿಗೆ [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಮತ್ತು [ಎಮಿ ಬೊಯ್ಡ್](https://twitter.com/AmyKateNicho) ಅವರನ್ನು ಬರೆಯಲಾಗಿದೆ

"ನ್ಯಾಯತೆ ಮತ್ತು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್" ಒಲುಮೆಯೊಂದಿಗೆ [ტომೋಮಿ ಇಮುರಾ](https://twitter.com/girliemac) ಅವರಿಂದ ಬರೆಯಲಾಗಿದೆ

"ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ತಂತ್ರಗಳು" ಒಲುಮೆಯೊಂದಿಗೆ [ಜೆನ್ ಲೂಪರ್](https://twitter.com/jenlooper) ಮತ್ತು [ಕ್ರಿಸ್ ನೋರಿಂಗ್](https://twitter.com/softchris) ಅವರಿಂದ ಬರೆಯಲಾಗಿದೆ

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ತಡೆಯಿರಿ**:  
ಈ ದಸ್ತಾವೇಜು AI ಅನುವಾದ ಸೇವೆ [Co-op Translator](https://github.com/Azure/co-op-translator) ಬಳಸಿ ಅನುವಾದಿಸಲಾಗಿದೆ. ನಾವು ನಿರ್ದಿಷ್ಟತೆಗಾಗಿ ಪ್ರಯತ್ನಿಸುವದಾದರೂ, ದಯವಿಟ್ಟು ಗಮನಿಸಿ ಸ್ವಯಂಚಾಲಿತ ಅನುವಾದಗಳಲ್ಲಿ ತಪ್ಪುಗಳು ಅಥವಾ ಅಸತ್ಯತೆಗಳು ಇದ್ದಿರಬಹುದು. ಮೂಲ ದಸ್ತಾವೇಜು ಅದರ ಪ್ರಾಕೃತ ಭಾಷೆಯಲ್ಲಿ ಅಧಿಕೃತ ಮೂಲ ಎಂದು ಪರಿಗಣಿಸಬೇಕು. ಮುಖ್ಯ ಮಾಹಿತಿಗಾಗಿ, ವೃತ್ತಿಪರ ಮಾನವ ಅನುವಾದವನ್ನು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ. ಈ ಅನುವಾದ ಬಳಕೆಯಿಂದ ಉಂಟಾಗುವ ಯಾವುದೇ ಭ್ರಮೆಗಳಿಗೆ ಅಥವಾ ತಪ್ಪು ಅರಿವಿಗೆ ನಾವು ಹೊಣೆಗಾರರೋಲ್ಲ.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->