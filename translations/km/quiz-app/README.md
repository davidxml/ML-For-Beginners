# សំណួរប្រកួតប្រជែង

សំណួរប្រកួតប្រជែងទាំងនេះគឺជាសំណួរមុននិងក្រោយម៉ោងរៀនសម្រាប់មុខវិជ្ជា ML នៅ https://aka.ms/ml-beginners

## ការតំឡើងគម្រោង

```
npm install
```

### សមាសភាព និងបញ្ចូលឡើងវិញទាន់ពេលសម្រាប់ការអភិវឌ្ឍ

```
npm run serve
```

### សមាសភាព និងបង្ហាប់សម្រាប់ការផលិត

```
npm run build
```

### ពិនិត្យកំហុស និងជួសជុលឯកសារ

```
npm run lint
```

### ការប្តូរតំរៀបរចនាសម្ព័ន្ធ

មើល [ឯកសារយោង Configuration Reference](https://cli.vuejs.org/config/)។

អនុគ្រោះៈ អរគុណចំពោះកំណែដើមនៃកម្មវិធីសំណួរប្រកួតប្រជែងនេះ៖ https://github.com/arpan45/simple-quiz-vue

## ការចែកចាយទៅ Azure

នេះជាមគ្គុទេសក៍ជំហាន​ដើម្បីជួយអ្នកចាប់ផ្តើម៖

1. Fork ប្រភព GitHub
ធានាថា​កូដកម្មវិធីបណ្ដាញស្តាទិច​របស់អ្នកមាននៅក្នុងផ្ទាំងចែករំលែក GitHub របស់អ្នក។ Fork ប្រភពនេះ។

2. បង្កើតកម្មវិធីបណ្ដាញស្តាទិច Azure
- បង្កើត និង [គណនី Azure](http://azure.microsoft.com)
- ទៅកាន់ [ផតថល Azure](https://portal.azure.com) 
- ចុច “Create a resource” និងស្វែងរក “Static Web App”។
- ចុច “Create”។

3. កំណត់រចនាសម្ព័ន្ធកម្មវិធីបណ្ដាញស្តាទិច
- Basics: Subscription: ជ្រើសរើសកម្មវិធីជាវ Azure របស់អ្នក។
- Resource Group: បង្កើតក្រុមធនធានថ្មីឬប្រើក្រុមមានរួចហើយ។
- Name: ផ្ដល់ឈ្មោះសម្រាប់កម្មវិធីបណ្ដាញស្តាទិចរបស់អ្នក។
- Region: ជ្រើសរើសតំបន់ជិតអ្នកប្រើប្រាស់បំផុត។

- #### ព័ត៌មាន​អំពីការចែកចាយ:
- Source: ជ្រើសរើស “GitHub”។
- GitHub Account: អនុញ្ញាត​ឲ្យ Azureចូលប្រើគណនី GitHub របស់អ្នក។
- Organization: ជ្រើសរើសអង្គការរបស់ GitHub របស់អ្នក។
- Repository: ជ្រើសរើសប្រភពដែលមានកម្មវិធីបណ្ដាញស្តាទិចរបស់អ្នក។
- Branch: ជ្រើសរើសសាខាដែលអ្នកចង់ចែកចាយពី។

- #### ព័ត៌មាន​អំពីការបង្កើត:
- Build Presets: ជ្រើសរើសសែវភ្លើងដែលកម្មវិធីរបស់អ្នកត្រូវបានកសាងជាមួយ (ឧ. React, Angular, Vue, ល។)។
- App Location: បញ្ជាក់ថតឯកសារដែលមានកូដកម្មវិធីរបស់អ្នក (ឧ. / ប្រសិនបើវាជារបស់បឋម)។
- API Location: ប្រសិនបើមាន API, បញ្ជាក់ទីតាំងរបស់វា (ជាការជ្រើសរើស)។
- Output Location: បញ្ជាក់ថតឯកសារដែលផលិតផលសាងសង់ត្រូវបានបង្កើត (ឧ. build ឬ dist)។

4. ពិនិត្យឡើងវិញ និងបង្កើត
ពិនិត្យកំណត់រចនាសម្ព័ន្ធរបស់អ្នក ហើយចុច “Create”។ Azure នឹងតំឡើងធនធានដែលចាំបាច់ទាំងអស់ និងបង្កើតកម្មវិធីប្រតិបត្តិ GitHub ក្នុងប្រភពរបស់អ្នក។

5. កម្មវិធីប្រតិបត្តិ GitHub Actions
Azure នឹងបង្កើតជា៉វិញឯកសារកម្មវិធីប្រតិបត្តិ GitHub Actions ក្នុងប្រភពរបស់អ្នក (.github/workflows/azure-static-web-apps-<name>.yml)។ កម្មវិធីនេះនឹងគ្រប់គ្រងដំណើរការសាងសង់ និងចែកចាយ។

6. ការត្រួតពិនិត្យការចែកចាយ
ទៅកាន់ផ្ទាំង “Actions” ក្នុងប្រភព GitHub របស់អ្នក។
អ្នកគួរតែឃើញកម្មវិធីប្រតិបត្តិកំពុងរត់។ កម្មវិធីនេះនឹងសាងសង់ និងចែកចាយកម្មវិធីបណ្ដាញស្តាទិចរបស់អ្នកទៅ Azure។
ពេលកម្មវិធីសម្រេចចប់ កម្មវិធីរបស់អ្នកនឹងមានបញ្ចាំងនៅលើ URL Azure ដែលផ្តល់ឲ្យ។

### ឧទាហរណ៍ឯកសារកម្មវិធីប្រតិបត្តិ

នេះជាឧទាហរណ៍នៃអ្វីដែលឯកសារកម្មវិធីប្រតិបត្តិ GitHub Actions អាចមានដូចខាងក្រោម៖
name: Azure Static Web Apps CI/CD
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### ឯកសារជំនួយបន្ថែម
- [ឯកសារពី Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [ឯកសារពី GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ការផ្តល់ដាច់ខាត**៖
ឯកសារនេះត្រូវបានបកប្រែដោយប្រើសេវាកម្មបកប្រែ AI [Co-op Translator](https://github.com/Azure/co-op-translator)។ នៅពេលយើងខិតខំរកភាពត្រឹមត្រូវ សូមយល់ថាការបកប្រែដោយស្វ័យប្រវត្តិប្រហែលជាអាចមានកំហុស ឬការខកចិត្តខ្លះៗ។ ឯកសារដើមនៅក្នុងភាសាដើមគួរត្រូវបានចាត់ទុកថាជា מקורអនុញ្ញាត។ សម្រាប់ព័ត៌មានសំខាន់ៗ សូមណែនាំឱ្យប្រើការបកប្រែដោយអ្នកជំនាញមនុស្ស។ យើងមិនទទួលខុសត្រូវចំពោះការយល់ច្រឡំ ឬការបកប្រែខុសពីការប្រើប្រាស់ការបកប្រែនេះឡើយ។
<!-- CO-OP TRANSLATOR DISCLAIMER END -->