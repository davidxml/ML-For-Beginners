# បង្កើតកម្មវិធីវែបដើម្បីប្រើម៉ូដែល ML របស់អ្នក

នៅក្នុងផ្នែកនេះនៃមេរៀន អ្នកនឹងបានណែនាំអំពីប្រធានបទ ML ប្រើប្រាស់បាន៖ របៀបរក្សាទុកម៉ូដែល Scikit-learn របស់អ្នកជាឯកសារដែលអាចប្រើសម្រាប់ធ្វើការទស្សន៍ទាយនៅក្នុងកម្មវិធីវែប។ បន្ទាប់ពីម៉ូដែលត្រូវបានរក្សាទុក អ្នកនឹងរៀនរបៀបទៅប្រើវាក្នុងកម្មវិធីវែបដែលបានបង្កើតក្នុង Flask។ អ្នកនឹងបានបង្កើតម៉ូដែលដោយប្រើទិន្នន័យដែលទាក់ទងគ្នានឹងការមើលឃើញ UFO! បន្ទាប់មក អ្នកនឹងបង្កើតកម្មវិធីវែបមួយដែលនឹងអនុញ្ញាតឱ្យអ្នកបញ្ចូលចំនួនវិនាទីជាមួយតម្លៃបណ្តោយ និងអទិសដៅដើម្បីទស្សន៍ទាយប្រទេសណាដែលបានរាយការណ៍ថាវិញឃើញ UFO។

![ឧទ្ទិស UFO](../../../translated_images/km/ufo.9e787f5161da9d4d.webp)

រូបថតដោយ <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> នៅ <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## មេរៀន

1. [បង្កើតកម្មវិធីវែប](1-Web-App/README.md)

## ការទទួលស្គាល់

"បង្កើតកម្មវិធីវែប" ត្រូវបានសរសេរដោយ ♥️ [Jen Looper](https://twitter.com/jenlooper)។

♥️ ការប្រឡងត្រូវបានសរសេរដោយ Rohan Raj។

ទិន្នន័យត្រូវបានយកមកពី [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)។

បែបផែនកម្មវិធីវែបត្រូវបានផ្តល់អនុសាសន៍ផ្នែកមួយដោយ [អត្ថបទនេះ](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) និង [repo នេះ](https://github.com/abhinavsagar/machine-learning-deployment) ដោយ Abhinav Sagar។

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ការបដិសេធ**៖  
ឯកសារនេះត្រូវបានបកប្រែដោយប្រើសេវាកម្មបកប្រែ AI [Co-op Translator](https://github.com/Azure/co-op-translator)។ ខណៈពេលដែលយើងខិតខំសំរាប់ភាពត្រឹមត្រូវ សូមយកចិត្តទុកដាក់ទៅលើការប្រែប្រើម៉ាស៊ីនដែលអាចមានកំហុស ឬភាពមិនត្រឹមត្រូវ។ ឯកសារដើមក្នុងភាសានិរន្តរបស់វាគួរត្រូវបានពិចារណាថាជាមូលដ្ឋានដែលមានអំណាច។ សម្រាប់ព័ត៌មានសំខាន់ៗ ការបកប្រែដោយមនុស្សជំនាញត្រូវបានផ្តល់អនុសាសន៍។ យើងមិនទទួលខុសត្រូវចំពោះការយល់ខុស ឬការបកប្រែខុសដោយបានប្រើប្រាស់ការបកប្រែនេះឡើយ។
<!-- CO-OP TRANSLATOR DISCLAIMER END -->