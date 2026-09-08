# សៀវភៅ​ជួយ​ដោះស្រាយ​បញ្ហា

សៀវភៅ​នេះ​ជួយ​អ្នក​ដោះស្រាយ​បញ្ហាទូទៅ​ពេលធ្វើការជាមួយ​មេរៀន Machine Learning សម្រាប់​អ្នក​ចាប់ផ្ដើម។ ប្រសិន​បើ​អ្នក​មិន​ស្វែងរក​ដំណោះស្រាយ​នៅទីនេះ​បាន សូម​ពិនិត្យមើល [ការពិភាក្សា Discord](https://aka.ms/foundry/discord) របស់យើង ឬ [បើកបញ្ហា](https://github.com/microsoft/ML-For-Beginners/issues)។

## តារាង​ខ្លឹម​សារ

- [បញ្ហា​ដំឡើងកម្មវិធី](#បញ្ហា​ដំឡើងកម្មវិធី)
- [បញ្ហា Jupyter Notebook](#ដំណោះស្រាយ-codeblock4)
- [បញ្ហា​កញ្ចប់ Python](#បញ្ហា​ក្រឡាចត្រង្គ-notebook)
- [បញ្ហា​បរិយាកាស R](#បញ្ហា​បញ្ចូលទិន្នន័យ)
- [បញ្ហា​កម្មវិធី Quiz](#ការដំឡើង​កញ្ចប់)
- [បញ្ហា​ទិន្នន័យ និង ផ្លូវ​ឯកសារ](#បញ្ហា​កម្មវិធី-quiz)
- [សារ​ខុស​ប្លែកៗ](#ដំណោះស្រាយ-codeblock18)
- [បញ្ហា​ប្រតិបត្តិការ](#កំហុស​អង្គចងចាំ)
- [បរិយាកាស និង ការកំណត់​រចនា](#កំហុស-unicodeencoding)

---

## បញ្ហា​ដំឡើងកម្មវិធី

### ការដំឡើង Python

**បញ្ហា**: `python: command not found`

**ដំណោះស្រាយ**:
1. ដំឡើង Python 3.8 ឬ​ខ្ពស់ជាង​ពី [python.org](https://www.python.org/downloads/)
2. ពិនិត្យ​ការ​ដំឡើង៖ `python --version` ឬ `python3 --version`
3. នៅលើ macOS/Linux អ្នកប្រហែលជា​ត្រូវប្រើ `python3` ជំនួស `python`

**បញ្ហា**: កំណែក Python ច្រើនបង្កប្រឈម

**ដំណោះស្រាយ**:
```bash
# ប្រើបរិយាកាសវីរុចជាចរាចរ ដើម្បីបំបែកគម្រោង
python -m venv ml-env

# បើកបរិយាកាសវីរុច
# លើ Windows:
ml-env\Scripts\activate
# លើ macOS/Linux:
source ml-env/bin/activate
```

### ការដំឡើង Jupyter

**បញ្ហា**: `jupyter: command not found`

**ដំណោះស្រាយ**:
```bash
# តំឡើង Jupyter
pip install jupyter

# ឬជាមួយ pip3
pip3 install jupyter

# ពិនិត្យការតំឡើង
jupyter --version
```

**បញ្ហា**: Jupyter មិនចាប់ផ្ដើម​ក្នុង​កម្មវិធីរុករក

**ដំណោះស្រាយ**:
```bash
# ព្យាយាមបញ្ជាក់កម្មវិធីរកមើលវេបសាយ
jupyter notebook --browser=chrome

# ឬចម្លង URL ជាមួយនឹង token ពី terminal ហើយបិទវាទៅកម្មវិធីរកមើលវេបសាយដោយដៃ
# ស្វែងរកៈ http://localhost:8888/?token=...
```

### ការដំឡើង R

**បញ្ហា**: កញ្ចប់ R មិនអាចដំឡើងបាន

**ដំណោះស្រាយ**:
```r
# ធ្វើឲ្យប្រាកដថាអ្នកមានកំណែ R ថ្មីបំផុត
# ដំឡើងកញ្ចប់ជាមួយនឹងការពឹងផ្អែក
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# ប្រសិនបើការប្រមូលកូដបរាជ័យ សូមព្យាយាមដំឡើងកំណែប៊ីនុារី
install.packages("package-name", type = "binary")
```

**បញ្ហា**: IRkernel មិនមាន​ក្នុង Jupyter

**ដំណោះស្រាយ**:
```r
# នៅក្នុងកុងសូល R
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## បញ្ហា Jupyter Notebook

### បញ្ហា Kernel

**បញ្ហា**: Kernel បន្តស្លាប់ឬចាប់ផ្ដើមឡើងវិញ

**ដំណោះស្រាយ**:
1. ចាប់ផ្ដើម kernel ថ្មី៖ `Kernel → Restart`
2. ខ្លះលទ្ធផលនិងចាប់ផ្ដើមឡើងវិញ៖ `Kernel → Restart & Clear Output`
3. ពិនិត្យបញ្ហា​អង្គចងចាំ (មើល [បញ្ហាប្រតិបត្តិការ](#កំហុស​អង្គចងចាំ))
4. ព្យាយាម​រត់ក្រឡាចត្រង្គ​តែបន្ទាត់ដើម្បីរកកូដបញ្ហា

**បញ្ហា**: ជ្រើស kernel Python មិនត្រឹមត្រូវ

**ដំណោះស្រាយ**:
1. ពិនិត្យ kernel បច្ចុប្បន្ន៖ `Kernel → Change Kernel`
2. ជ្រើសកំណែ Python ត្រឹមត្រូវ
3. ប្រសិន kernel មិនមាន បង្កើតវា៖
```bash
python -m ipykernel install --user --name=ml-env
```

**បញ្ហា**: Kernel មិនចាប់ផ្ដើម

**ដំណោះស្រាយ**:
```bash
# ធ្វើការដំឡើង ipykernel ម្តងទៀត
pip uninstall ipykernel
pip install ipykernel

# ចុះបញ្ជី kernel ម្តងទៀត
python -m ipykernel install --user
```

### បញ្ហា​ក្រឡាចត្រង្គ Notebook

**បញ្ហា**: ក្រឡាចត្រង្គ​កំពុងរត់ប៉ុន្តែមិនបង្ហាញលទ្ធផល

**ដំណោះស្រាយ**:
1. ពិនិត្យមើលថា​ក្រឡាចត្រង្គ​នៅតែ​រត់ (មើលបំណែក `[*]`)
2. ចាប់ផ្ដើម kernel ថ្មីនិង​រត់​ក្រឡាចត្រង្គ​ទាំងអស់៖ `Kernel → Restart & Run All`
3. ពិនិត្យ console របស់ក្រុមហ៊ុន​រុករកសម្រាប់កំហុស JavaScript (F12)

**បញ្ហា**: មិនអាច​រត់​ក្រឡាចត្រង្គ​បាន - មិនមាន ප්ទិសការឆ្លើយតបពេលចុច "Run"

**ដំណោះស្រាយ**:
1. ពិនិត្យមើលថា​សេវាកម្ម Jupyter នៅតែ​ដំណើរការ​នៅក្នុង terminal
2. ហៅទំព័ររុករកឡើងវិញ
3. បិទហើយបើកឡើងវិញ notebook
4. ចាប់ផ្ដើម​សេវាកម្ម Jupyter ថ្មី

---

## បញ្ហា​កញ្ចប់ Python

### កំហុស Import

**បញ្ហា**: `ModuleNotFoundError: No module named 'sklearn'`

**ដំណោះស្រាយ**:
```bash
pip install scikit-learn

# កញ្ចប់ ML ទូទៅសម្រាប់វគ្គនេះ
pip install scikit-learn pandas numpy matplotlib seaborn
```

**បញ្ហា**: `ImportError: cannot import name 'X' from 'sklearn'`

**ដំណោះស្រាយ**:
```bash
# ធ្វើបច្ចុប្បន្នភាព scikit-learn ទៅកំណែក្រឡាប់ថ្មីបំផុត
pip install --upgrade scikit-learn

# ពិនិត្យកំណែ
python -c "import sklearn; print(sklearn.__version__)"
```

### បញ្ហា​កំណែ​រំខាន

**បញ្ហា**: កំហុស​អត្រាកញ្ចប់​កំណែ​មិនសម

**ដំណោះស្រាយ**:
```bash
# បង្កើតបរិយាកាសប្រព័ន្ធមេនវីឌួ
python -m venv fresh-env
source fresh-env/bin/activate  # ឬ fresh-env\Scripts\activate នៅលើ Windows

# ដំឡើងកញ្ចប់ថ្មី
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# ប្រសិនបើត្រូវការកំណែជាក់លាក់
pip install scikit-learn==1.3.0
```

**បញ្ហា**: `pip install` បរាជ័យដោយកំហុស​សិទ្ធិ

**ដំណោះស្រាយ**:
```bash
# ដំឡើងសម្រាប់អ្នកប្រើបច្ចុប្បន្នតែប៉ុណ្ណោះ
pip install --user package-name

# ឬប្រើបរិយាកាសវឌ្ឍនបត Virtual (សំណូមពរ)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### បញ្ហា​បញ្ចូលទិន្នន័យ

**បញ្ហា**: កំហុស `FileNotFoundError` ពេល​បញ្ចូល​ឯកសារ CSV

**ដំណោះស្រាយ**:
```python
import os
# ពិនិត្យទីតាំងការងារបច្ចុប្បន្ន
print(os.getcwd())

# ប្រើផ្លូវបញ្ជាមួយទាក់ទងពីទីតាំងសៀវភៅកំណត់ហេតុនេះ
df = pd.read_csv('../../data/filename.csv')

# ឬប្រើផ្លូវបញ្ជាពេញលេញ
df = pd.read_csv('/full/path/to/data/filename.csv')
```

---

## បញ្ហា​បរិយាកាស R

### ការដំឡើង​កញ្ចប់

**បញ្ហា**: ការដំឡើង​កញ្ចប់​បរាជ័យដោយកំហុស​កម្មង់

**ដំណោះស្រាយ**:
```r
# តំឡើងជាកំណែទ្វាពីរប្រព័ន្ធ (Windows/macOS)
install.packages("package-name", type = "binary")

# ធ្វើបច្ចុប្បន្នភាព R ទៅកំណែចុងក្រោយ ប្រសិនបើកញ្ចប់ត្រូវការ
# ពិនិត្យកំណែ R
R.version.string

# តំឡើងភាពពឹងផ្អែករបស់ប្រព័ន្ធ (Linux)
# សម្រាប់ Ubuntu/Debian, នៅក្នុងផ្ទាំងពាក្យបញ្ជា:
# sudo apt-get install r-base-dev
```

**បញ្ហា**: `tidyverse` មិនដំឡើង

**ដំណោះស្រាយ**:
```r
# ដំឡើងការពឹងផ្អែកជាដំបូង
install.packages(c("rlang", "vctrs", "pillar"))

# បន្ទាប់មកដំឡើង tidyverse
install.packages("tidyverse")

# រឺដំឡើងធាតុផ្សំម្នាក់ៗ
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### បញ្ហា RMarkdown

**បញ្ហា**: RMarkdown មិនបង្ហាញលទ្ធផល

**ដំណោះស្រាយ**:
```r
# ដំឡើង/បន្ទាន់សម័យ rmarkdown
install.packages("rmarkdown")

# ដំឡើង pandoc ប្រសិនបើចាំបាច់
install.packages("pandoc")

# សម្រាប់លទ្ធផល PDF, ដំឡើង tinytex
install.packages("tinytex")
tinytex::install_tinytex()
```

---

## បញ្ហា​កម្មវិធី Quiz

### ការសាងសង់ និង ដំឡើង

**បញ្ហា**: `npm install` បរាជ័យ

**ដំណោះស្រាយ**:
```bash
# លុបឃ្លាំង npm
npm cache clean --force

# លុប node_modules និង package-lock.json
rm -rf node_modules package-lock.json

# ដំឡើងឡើងវិញ
npm install

# ប្រសិនបើមិនបានសូមព្យាយាមជាមួយ legacy peer deps
npm install --legacy-peer-deps
```

**បញ្ហា**: ស្វ័យប្រវត្តិ Port 8080 កំពុងប្រើ

**ដំណោះស្រាយ**:
```bash
# ប្រើពួតផ្សេង
npm run serve -- --port 8081

# រឺស្វែងរកនិងបញ្ឈប់ដំណើរការប្រើពួត 8080
# នៅលើ Linux/macOS:
lsof -ti:8080 | xargs kill -9

# នៅលើ Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### កំហុសសាងសង់

**បញ្ហា**: `npm run build` បរាជ័យ

**ដំណោះស្រាយ**:
```bash
# ពិនិត្យជំនាន់ Node.js (គួរតែលើស 14)
node --version

# បន្ទាន់សម័យ Node.js ប្រសិនបើចាំបាច់
# បន្ទាប់មកធ្វើការតម្លើងថ្មី
rm -rf node_modules package-lock.json
npm install
npm run build
```

**បញ្ហា**: កំហុស linting បង្កការពិបាកសាងសង់

**ដំណោះស្រាយ**:
```bash
# ដោះស្រាយបញ្ហាដែលអាចជួសជុលដោយស្វ័យប្រវត្តិ
npm run lint -- --fix

# ឬបិទសកម្មភាពលីនថ៍បណ្ដោះអាសន្ននៅក្នុងការចាក់បញ្ចាំង
# (មិនផ្ដល់អនុសាសន៍សម្រាប់ផលិតកម្ម)
```

---

## បញ្ហា​ទិន្នន័យ និង ផ្លូវ​ឯកសារ

### បញ្ហាផ្លូវ

**បញ្ហា**: ឯកសារទិន្នន័យមិនត្រូវបានរកឃើញពេល​រត់ notebook

**ដំណោះស្រាយ**:
1. **តែងតែ​រត់ notebook ពីថត​ដែលវាស្ថិត​នៅក្នុង**  
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **ពិនិត្យ​ផ្លូវទ relatif នៅក្នុងកូដ**  
   ```python
   # ផ្លូវត្រឹមត្រូវពីទីតាំងសៀវភៅកំណត់ត្រា
   df = pd.read_csv('../data/filename.csv')
   
   # មិនមកពីទីតាំងប្រដាប់បញ្ជារបស់អ្នកទេ
   ```

3. **ប្រើផ្លូវ​លេខស្មើ (absolute paths) ប្រសិនបើចាំបាច់**  
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### ឯកសារ​ទិន្នន័យ​អវសាន

**បញ្ហា**: ឯកសារតំណាងទិន្នន័យបាត់បង់

**ដំណោះស្រាយ**:
1. ពិនិត្យមើលថាតើទិន្នន័យ​គួរតែ​មាន​ក្នុង repository ​- រាល់ទិន្នន័យភាគច្រើនបានរួមបញ្ចូល
2. មេរៀនខ្លះត្រូវការទាញយកទិន្នន័យ - ពិនិត្យមើល README មេរៀន
3. ​ប្រាកដថាអ្នកបានទាញយកការផ្លាស់ប្តូរថ្មីៗ៖  
   ```bash
   git pull origin main
   ```

---

## សារ​ខុស​ប្លែកៗ

### កំហុស​អង្គចងចាំ

**កំហុស**: `MemoryError` ឬ kernel ស្លាប់ពេល​ដំណើរការ​ទិន្នន័យ

**ដំណោះស្រាយ**:
```python
# បញ្ចូលទិន្នន័យជាមួយប្លុក
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# ឬអានព្រឹត្តិការណ៍តែលេខ្វាងទេ
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# សម្លឹងចេញពីម៉ឺម៉ូរីពេលបញ្ចប់
del large_dataframe
import gc
gc.collect()
```

### ការព្រមាន Convergence

**ព្រមាន**: `ConvergenceWarning: Maximum number of iterations reached`

**ដំណោះស្រាយ**:
```python
from sklearn.linear_model import LogisticRegression

# បង្កើនចំនួន iteration អតិបរមា
model = LogisticRegression(max_iter=1000)

# ឬបង្ហាញលក្ខណៈរបស់អ្នកជាមុនសិន
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### បញ្ហាផ្ទាំងក្រាហ្វិក

**បញ្ហា**: ផ្ទាំងក្រាហ្វិកមិនបង្ហាញក្នុង Jupyter

**ដំណោះស្រាយ**:
```python
# បើកការរៀបចំគំនូសបន្ទាត់
%matplotlib inline

# នាំចូល pyplot
import matplotlib.pyplot as plt

# បង្ហាញគំនូសយ៉ាងច្បាស់
plt.plot(data)
plt.show()
```

**បញ្ហា**: ផ្ទាំងក្រាហ្វិក Seaborn មើលខុសឬបង្ហាញកំហុស

**ដំណោះស្រាយ**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# បន្ទាន់សម័យទៅកំណែសមត្ថភាព
# pip install --upgrade seaborn matplotlib
```

### កំហុស Unicode/Encoding

**បញ្ហា**: `UnicodeDecodeError` ពេល​អានឯកសារ

**ដំណោះស្រាយ**:
```python
# បញ្ជាក់កូដបញ្ចូលឲ្យច្បាស់
df = pd.read_csv('file.csv', encoding='utf-8')

# ឬសាកល្បងកូដបញ្ចូលផ្សេងទៀត
df = pd.read_csv('file.csv', encoding='latin-1')

# សម្រាប់ errors='ignore' ដើម្បីរំលងតួអក្សរដែលមានបញ្ហា
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## បញ្ហា​ប្រតិបត្តិការ

### ការរត់ notebook យឺត

**បញ្ហា**: Notebook រត់យឺតខ្លាំង

**ដំណោះស្រាយ**:
1. **ចាប់ផ្ដើម kernel ថ្មី ដើម្បីធ្វើអង្គចងចាំមូល**៖ `Kernel → Restart`  
2. **បិទ notebook មិនប្រើ** ដើម្បីសង្រ្គោះធនធាន  
3. **ប្រើតំណាងទិន្នន័យតូចសម្រាប់សាកល្បង**:  
   ```python
   # ធ្វើការ​ជាមួយ​ផ្នែក​តូច​ក្នុងអំឡុងពេលអភិវឌ្ឍន៍
   df_sample = df.sample(n=1000)
   ```
4. **រាយការណ៍កម្មវិធីរបស់អ្នក** ដើម្បីរកកន្លែងដាក់ពេលយឺត:  
   ```python
   %time operation()  # ពេលវេលាសម្រាប់ប្រតិបត្តិការតែមួយ
   %timeit operation()  # ពេលវេលាជាមួយការរត់ច្រើនដង
   ```

### ការប្រើប្រាស់អង្គចងចាំខ្ពស់

**បញ្ហា**: ប្រព័ន្ធ​ប្រើអង្គចងចាំ​ច្រើនពេក

**ដំណោះស្រាយ**:
```python
# ពិនិត្យមើលការប្រើប្រាស់អង្គចងចាំ
df.info(memory_usage='deep')

# បង្កើនប្រសិទ្ធភាពប្រភេទទិន្នន័យ
df['column'] = df['column'].astype('int32')  # ជំនួស int64

# ទម្លាក់ជួរឈរដែលមិនចាំបាច់
df = df[['col1', 'col2']]  # រក្សាទុកតែជួរឈរដែលចាំបាច់ប៉ុណ្ណោះ

# ដំណើរការជាក្រុម
for batch in np.array_split(df, 10):
    process(batch)
```

---

## បរិយាកាស និង ការកំណត់​រចនា

### បញ្ហាបរិយាកាស Virtual

**បញ្ហា**: បរិយាកាស Virtual មិនដំណើរការ

**ដំណោះស្រាយ**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# ពិនិត្យ​មើល​ថា​តើ​បានបើក​ប្រើ​ហើយ​ឬនៅ (គួរ​តែបង្ហាញឈ្មោះ venv នៅក្នុង prompt)
which python  # គួរតែបង្ហាញ python របស់ venv
```

**បញ្ហា**: កញ្ចប់​ដំឡើងហើយប៉ុន្តែមិនរកឃើញ​ក្នុង notebook

**ដំណោះស្រាយ**:
```bash
# ប្រាកដថា notebook ប្រើ kernel ត្រឹមត្រូវ
# តំឡើង ipykernel នៅក្នុង venv របស់អ្នក
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# នៅក្នុង Jupyter: Kernel → ប្ដូរ Kernel → Python (ml-env)
```

### បញ្ហា Git

**បញ្ហា**: មិនអាចទាញយកកំណែថ្មី - ប្រឈមមុខកំហុសបញ្ចូល​ (merge conflicts)

**ដំណោះស្រាយ**:
```bash
# រក្សាទុកការផ្លាស់ប្តូររបស់អ្នក
git stash

# ទាញយកចុងក្រោយ
git pull origin main

# អនុវត្តបម្លែងរបស់អ្នកម្តងទៀត
git stash pop

# ប្រសិនបើមានជម្លោះ សូមដោះស្រាយដោយដៃ ឬ៖
git checkout --theirs path/to/file  # ទទួលយកកំណែពីចម្ងាយ
git checkout --ours path/to/file    # រក្សាកំណែរបស់អ្នក
```

### ការបញ្ចូល VS Code

**បញ្ហា**: Jupyter notebook មិនបើកក្នុង VS Code

**ដំណោះស្រាយ**:
1. ដំឡើង ផ្នែកបន្ថែម Python ក្នុង VS Code  
2. ដំឡើង ផ្នែកបន្ថែម Jupyter ក្នុង VS Code  
3. ជ្រើសកំណែ Python ត្រឹមត្រូវ៖ `Ctrl+Shift+P` → "Python: Select Interpreter"  
4. ចាប់ផ្ដើម VS Code ថ្មី  

---

## ឯកសារជំនួយ​បន្ថែម

- **Discord Discussions**: [សួរជម្លើយ និងចែករំលែកដំណោះស្រាយនៅក្នុងបន្ទប់ #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ម៉ូឌុល ML សម្រាប់អ្នកចាប់ផ្ដើម](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **មេរៀន​វីដេអូ**: [បញ្ជីផ្តល់មេរៀន YouTube](https://aka.ms/ml-beginners-videos)
- **តាមដានបញ្ហា**: [រាយការណ៍កំហុស](https://github.com/microsoft/ML-For-Beginners/issues)

---

## តើ​អ្នកនៅតែ​មាន​បញ្ហា?

បើអ្នកបាន​ព្យាយាមដំណោះស្រាយខាងលើហើយក៏នៅតែ​មានបញ្ហា៖

1. **ស្វែងរកបញ្ហាមុនមានរួច**៖ [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)  
2. **ពិនិត្យការពិភាក្សា​នៅ Discord**៖ [Discord Discussions](https://aka.ms/foundry/discord)  
3. **បើកបញ្ហាថ្មី**៖ រួមបញ្ចូល៖  
   - ប្រព័ន្ធ​ប្រតិបត្តិការ និងកំណែ​របស់អ្នក  
   - កំណែ Python/R  
   - សារខុស (តាមការតាមដានពេញលេញ)  
   - ជំហាន​ផលិតបញ្ហា  
   - អ្វីដែលអ្នកបានព្យាយាម​រួចហើយ  

យើងនៅទីនេះដើម្បីជួយអ្នក! 🚀

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ការបដិសេធ** ៖  
ឯកសារនេះត្រូវបានបកប្រែដោយប្រើសេវាកម្មបកប្រែ AI [Co-op Translator](https://github.com/Azure/co-op-translator)។ ខណៈពេលយើងខិតខំរកភាពត្រឹមត្រូវ សូមយល់ដល់ថាការបកប្រែដោយស្វ័យប្រវត្តិក្នុងមួយពេលអាចមានកំហុសឬភាពមិនត្រឹមត្រូវ។ ឯកសារដើមដែលនៅក្នុងភាសាមាតុភូមិគួរត្រូវបានពិចារណាថាជា ប្រភពតែមួយដែលមានសុពលភាព។ សម្រាប់ព័ត៌មានសំខាន់ៗ គឺត្រូវបានណែនាំឲ្យប្រើការបកប្រែដោយមនុស្សអ្នកជំនាញ។ យើងមិនមានកាតព្វកិច្ចរំលោភចំពោះការយល់ច្រឡំ ឬការបកប្រែខុសក្នុងការប្រើប្រាស់បកប្រែនេះឡើយ។
<!-- CO-OP TRANSLATOR DISCLAIMER END -->