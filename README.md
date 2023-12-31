# Description
Exploring multi-class classification with CVE paragraph texts as input and CVSS vectors C, I and A as labels with discrete values each having either none, low or high.
In this POC, only the `confidentiality` vector is trained to the model to predict the `confidentiality` vector of a vulnerabiltiy description text.

# How to Run
1. Install the necessary pre-requisites using `pipenv install`
2. Then `pipenv shell` to load the virtual environment
3. To train, run `python confidentiality_train.py`
4. To predict, run `python confidentiality_predict.py`
5. Feel free to tweak the hyper parameters within the `confidentiality_train.py`
6. To load your data, feel free to write a simple CSV routine or just (lazy like me) use the `training_data.py``

## Training example
Please note that training accuracy is never realistically at 1.0 (or 100%) and even if it's possible, it's overfitting (not an ideal outcome). Due to the lack of data,  
```
[*] Welcome to text classification multi-class model training!
[$] Hyper Parameters:
    Num of Epochs: 15
    Batch Size: 8
    TEXT_MAX_LENGTH: 256
    LEARNING_RATE: 1e-05
[*] Training initializing...
[+] Available device for training: -> cuda
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[*] Total number of training records: 71
/home/js/.local/share/virtualenvs/explore-multi-label-classification-BnqXW7ng/lib/python3.11/site-packages/transformers/optimization.py:411: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
[*] Epoch 1/15, Training loss: 1.2126 | Training accuracy: 0.1786 | Time elapsed: 1.85 sec
[*] Epoch 2/15, Training loss: 1.0356 | Training accuracy: 0.3929 | Time elapsed: 0.81 sec
[*] Epoch 3/15, Training loss: 0.9558 | Training accuracy: 0.4821 | Time elapsed: 0.76 sec
[*] Epoch 4/15, Training loss: 0.9398 | Training accuracy: 0.5357 | Time elapsed: 0.77 sec
[*] Epoch 5/15, Training loss: 0.8988 | Training accuracy: 0.4821 | Time elapsed: 0.76 sec
[*] Epoch 6/15, Training loss: 0.9294 | Training accuracy: 0.5000 | Time elapsed: 0.78 sec
[*] Epoch 7/15, Training loss: 0.9285 | Training accuracy: 0.5000 | Time elapsed: 0.82 sec
[*] Epoch 8/15, Training loss: 0.8785 | Training accuracy: 0.5714 | Time elapsed: 0.81 sec
[*] Epoch 9/15, Training loss: 0.8784 | Training accuracy: 0.6786 | Time elapsed: 0.81 sec
[*] Epoch 10/15, Training loss: 0.8598 | Training accuracy: 0.6964 | Time elapsed: 0.83 sec
[*] Epoch 11/15, Training loss: 0.7776 | Training accuracy: 0.7679 | Time elapsed: 0.82 sec
[*] Epoch 12/15, Training loss: 0.7464 | Training accuracy: 0.8393 | Time elapsed: 0.81 sec
[*] Epoch 13/15, Training loss: 0.7234 | Training accuracy: 0.7321 | Time elapsed: 0.82 sec
[*] Epoch 14/15, Training loss: 0.6875 | Training accuracy: 0.8571 | Time elapsed: 0.83 sec
[*] Epoch 15/15, Training loss: 0.6165 | Training accuracy: 0.9107 | Time elapsed: 0.82 sec
[*] Total elapsed time: 13.10 sec
[+] Training completed with total Epoch: 15 - Total loss: 0.5359 - Elapsed Time: 13.10
```

## Prediction Example
These input text are real CVE descriptions from cvedetails.com. Using the pseudo training data (generated by Chatgpt 3.5), it seems to work well although these pseudo training data was meant to validate the model in an "ideal" situation. 
```
Input Text: Jenkins Docker Swarm Plugin 1.11 and earlier does not escape values returned from Docker before inserting them into the Docker Swarm Dashboard view, resulting in a stored cross-site scripting (XSS) vulnerability exploitable by attackers able to control responses from Docker.
Predicted CVSS Confidentiality: low - tensor([0.0201, 0.9162, 0.0637], grad_fn=<SelectBackward0>)
================================================================================
Input Text: Multiple reflected XSS were found on different JSP files with unsanitized parameters in OpenMNS Horizon 31.0.8 and versions earlier than 32.0.2 on multiple platforms that an attacker can modify to craft a malicious XSS payload. The solution is to upgrade to Meridian 2023.1.6, 2022.1.19, 2021.1.30, 2020.1.38 or Horizon 32.0.2 or newer.
Predicted CVSS Confidentiality: low - tensor([0.0210, 0.9219, 0.0571], grad_fn=<SelectBackward0>)
================================================================================
Input Text: WebBoss.io CMS v3.7.0.1 contains a stored cross-site scripting (XSS) vulnerability.
Predicted CVSS Confidentiality: low - tensor([0.0175, 0.9370, 0.0455], grad_fn=<SelectBackward0>)
================================================================================
Input Text: A reflected cross-site scripting (XSS) vulnerability in the component /ui/diagnostics/log/core/ of OPNsense before 23.7 allows attackers to inject arbitrary JavaScript via the URL path.
Predicted CVSS Confidentiality: low - tensor([0.0231, 0.9285, 0.0485], grad_fn=<SelectBackward0>)
================================================================================
Input Text: GNU inetutils through 2.4 may allow privilege escalation because of unchecked return values of set*id() family functions in ftpd, rcp, rlogin, rsh, rshd, and uucpd. This is, for example, relevant if the setuid system call fails when a process is trying to drop privileges before letting an ordinary user control the activities of the process.
Predicted CVSS Confidentiality: high - tensor([0.0518, 0.1844, 0.7638], grad_fn=<SelectBackward0>)
================================================================================
Input Text: Cryptomator encrypts data being stored on cloud infrastructure. The MSI installer provided on the homepage for Cryptomator version 1.9.2 allows local privilege escalation for low privileged users, via the `repair` function. The problem occurs as the repair function of the MSI is spawning an SYSTEM Powershell without the `-NoProfile` parameter. Therefore the profile of the user starting the repair will be loaded. Version 1.9.3 contains a fix for this issue. Adding a `-NoProfile` to the powershell is a possible workaround.
Predicted CVSS Confidentiality: high - tensor([0.0525, 0.3742, 0.5733], grad_fn=<SelectBackward0>)
================================================================================
Input Text: Vulnerability of API privilege escalation in the wifienhance module. Successful exploitation of this vulnerability may cause the arp list to be modified.
Predicted CVSS Confidentiality: high - tensor([0.0445, 0.2070, 0.7485], grad_fn=<SelectBackward0>)
================================================================================
Input Text: Permission control vulnerability in the audio module. Successful exploitation of this vulnerability may cause audio devices to perform abnormally.
Predicted CVSS Confidentiality: high - tensor([0.0923, 0.1171, 0.7906], grad_fn=<SelectBackward0>)
================================================================================
Input Text: ImageMagick before 6.9.12-91 allows attackers to cause a denial of service (memory consumption) in Magick::Draw.
Predicted CVSS Confidentiality: none - tensor([0.6359, 0.1770, 0.1872], grad_fn=<SelectBackward0>)
================================================================================
Input Text: Improper input validation in Zoom SDK’s before 5.14.10 may allow an unauthenticated user to enable a denial of service via network access.
Predicted CVSS Confidentiality: none - tensor([0.6308, 0.2013, 0.1678], grad_fn=<SelectBackward0>)
================================================================================
Input Text: Heap buffer overflow in paddle.trace in PaddlePaddle before 2.5.0. This flaw can lead to a denial of service, information disclosure, or more damage is possible.
Predicted CVSS Confidentiality: none - tensor([0.6283, 0.1748, 0.1968], grad_fn=<SelectBackward0>)
================================================================================
Input Text: IBM WebSphere Application Server Liberty 22.0.0.13 through 23.0.0.7 is vulnerable to a denial of service, caused by sending a specially-crafted request. A remote attacker could exploit this vulnerability to cause the server to consume memory resources. IBM X-Force ID: 262567.
Predicted CVSS Confidentiality: none - tensor([0.6161, 0.1754, 0.2085], grad_fn=<SelectBackward0>)
================================================================================
Input Text: iCMS v7.0.16 was discovered to contain a SQL injection vulnerability via the bakupdata function.
Predicted CVSS Confidentiality: high - tensor([0.0426, 0.1472, 0.8102], grad_fn=<SelectBackward0>)
================================================================================
Input Text: PrestaShop is an open source e-commerce web application. Prior to version 8.1.1, SQL injection possible in the product search field, in BO's product page. Version 8.1.1 contains a patch for this issue. There are no known workarounds.
Predicted CVSS Confidentiality: high - tensor([0.0557, 0.1923, 0.7520], grad_fn=<SelectBackward0>)
================================================================================
Input Text: webchess v1.0 was discovered to contain a SQL injection vulnerability via the $playerID parameter at mainmenu.php.
Predicted CVSS Confidentiality: low - tensor([0.0357, 0.5186, 0.4457], grad_fn=<SelectBackward0>)
================================================================================
Input Text: social-media-skeleton is an uncompleted social media project. A SQL injection vulnerability in the project allows UNION based injections, which indirectly leads to remote code execution. Commit 3cabdd35c3d874608883c9eaf9bf69b2014d25c1 contains a fix for this issue.
Predicted CVSS Confidentiality: high - tensor([0.0509, 0.2451, 0.7040], grad_fn=<SelectBackward0>)
```