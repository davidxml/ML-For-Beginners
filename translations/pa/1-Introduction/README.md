# ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਪਰੇਚੈ

ਇਸ ਸਿਲੇਬਸ ਦੇ ਹਿੱਸੇ ਵਿੱਚ, ਤੁਹਾਨੂੰ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੇ ਬੁਨਿਆਦੀ ਧਾਰણਾਵਾਂ ਨਾਲ ਜਾਣੂ ਕਰਵਾਇਆ ਜਾਵੇਗਾ, ਜਿਸ ਵਿੱਚ ਇਹ ਕੀ ਹੈ, ਇਸ ਦਾ ਇਤਿਹਾਸ, ਅਤੇ ਖੋਜਕਰਤਾ ਇਸਨੂੰ ਅਸਲੀ ਦੁਨੀਆਂ ਦੇ ਸੰਦਰਭਾਂ ਵਿੱਚ ਲਾਗੂ ਕਰਨ ਲਈ ਵਰਤਦੇ ਹਨ, ਸ਼ਾਮਲ ਹਨ। ਆਓ ਇਸ ਰੋਮਾਂਚਕ ML ਦੀ ਦੁਨੀਆ ਨੂੰ ਇਕੱਠੇ ਖੋਜੀਏ!

![globe](../../../translated_images/pa/globe.59f26379ceb40428.webp)
> ਫੋਟੋ <a href="https://unsplash.com/@bill_oxford?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਬਿੱਲ ਆਕਸਫੋਰਡ</a> ਵਲੋਂ <a href="https://unsplash.com/s/photos/globe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">ਅਨਸਪਲੈਸ਼</a> 'ਤੇ

### ਪਾਠ

1. [ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਪਰੇਚੈ](1-intro-to-ML/README.md)
1. [ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਅਤੇ ਏਆਈ ਦਾ ਇਤਿਹਾਸ](2-history-of-ML/README.md)
1. [ਨਿਆਇਕਤਾ ਅਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ](3-fairness/README.md)
1. [ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੀਆਂ ਤਕਨੀਕਾਂ](4-techniques-of-ML/README.md)
### ਸਰੋਕਾਰ

"ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਪਰੇਚੈ" ਪਿਆਰ ਨਾਲ ਇੱਕ ਟੀਮ ਵੱਲੋਂ ਲਿਖਿਆ ਗਿਆ ਜਿਸ ਵਿੱਚ ਸ਼ਾਮਲ ਹਨ [ਮੁਹੰਮਦ ਸਾਕਿਬ ਖਾਨ ਇਨਾਨ](https://twitter.com/Sakibinan), [ਓਰਨੈਲਾ ਅਲਤੁਨਯਾਨ](https://twitter.com/ornelladotcom) ਅਤੇ [ਜੇਨ ਲੂਪਰ](https://twitter.com/jenlooper)

"ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦਾ ਇਤਿਹਾਸ" ਪਿਆਰ ਨਾਲ [ਜੇਨ ਲੂਪਰ](https://twitter.com/jenlooper) ਅਤੇ [ਐਮੀ ਬੌਇਡ](https://twitter.com/AmyKateNicho) ਵਲੋਂ ਲਿਖਿਆ ਗਿਆ

"ਨਿਆਇਕਤਾ ਅਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ" ਪਿਆਰ ਨਾਲ [ਟੋਮੋਮੀ ਇਮુરਾ](https://twitter.com/girliemac) ਵਲੋਂ ਲਿਖਿਆ ਗਿਆ

"ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੀਆਂ ਤਕਨੀਕਾਂ" ਪਿਆਰ ਨਾਲ [ਜੇਨ ਲੂਪਰ](https://twitter.com/jenlooper) ਅਤੇ [ਕ੍ਰਿਸ ਨੋਰਿੰਗ](https://twitter.com/softchris) ਵਲੋਂ ਲਿਖਿਆ ਗਿਆ

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ਬ੍ਰਹਮਚਿੰਤਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਰੱਖੋ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸਤੀਤਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਆਪਣੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਹੀ ਜ਼ਰੂਰੀ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਣ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਿਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਅਸੀਂ ਇਸ ਅਨੁਵਾਦ ਦੇ ਉਪਯੋਗ ਤੋਂ ਉਤਪੰਨ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਮਤਲਬ ਲਈ ਜਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->