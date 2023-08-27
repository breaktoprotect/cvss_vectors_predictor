import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from training_data import X_train, cia_output


def main():
    # Real CVE
    input_texts = [
        # ? xss
        "Jenkins Docker Swarm Plugin 1.11 and earlier does not escape values returned from Docker before inserting them into the Docker Swarm Dashboard view, resulting in a stored cross-site scripting (XSS) vulnerability exploitable by attackers able to control responses from Docker.",
        "Multiple reflected XSS were found on different JSP files with unsanitized parameters in OpenMNS Horizon 31.0.8 and versions earlier than 32.0.2 on multiple platforms that an attacker can modify to craft a malicious XSS payload. The solution is to upgrade to Meridian 2023.1.6, 2022.1.19, 2021.1.30, 2020.1.38 or Horizon 32.0.2 or newer.",
        "WebBoss.io CMS v3.7.0.1 contains a stored cross-site scripting (XSS) vulnerability.",
        "A reflected cross-site scripting (XSS) vulnerability in the component /ui/diagnostics/log/core/ of OPNsense before 23.7 allows attackers to inject arbitrary JavaScript via the URL path.",
        # ? priv esc
        "GNU inetutils through 2.4 may allow privilege escalation because of unchecked return values of set*id() family functions in ftpd, rcp, rlogin, rsh, rshd, and uucpd. This is, for example, relevant if the setuid system call fails when a process is trying to drop privileges before letting an ordinary user control the activities of the process.",
        "Cryptomator encrypts data being stored on cloud infrastructure. The MSI installer provided on the homepage for Cryptomator version 1.9.2 allows local privilege escalation for low privileged users, via the `repair` function. The problem occurs as the repair function of the MSI is spawning an SYSTEM Powershell without the `-NoProfile` parameter. Therefore the profile of the user starting the repair will be loaded. Version 1.9.3 contains a fix for this issue. Adding a `-NoProfile` to the powershell is a possible workaround.",
        "Vulnerability of API privilege escalation in the wifienhance module. Successful exploitation of this vulnerability may cause the arp list to be modified.",
        "Permission control vulnerability in the audio module. Successful exploitation of this vulnerability may cause audio devices to perform abnormally.",
        # ? DoS
        "ImageMagick before 6.9.12-91 allows attackers to cause a denial of service (memory consumption) in Magick::Draw.",
        "Improper input validation in Zoom SDK’s before 5.14.10 may allow an unauthenticated user to enable a denial of service via network access.",
        "Heap buffer overflow in paddle.trace in PaddlePaddle before 2.5.0. This flaw can lead to a denial of service, information disclosure, or more damage is possible.",
        "IBM WebSphere Application Server Liberty 22.0.0.13 through 23.0.0.7 is vulnerable to a denial of service, caused by sending a specially-crafted request. A remote attacker could exploit this vulnerability to cause the server to consume memory resources. IBM X-Force ID: 262567.",
        # ? SQLi
        "iCMS v7.0.16 was discovered to contain a SQL injection vulnerability via the bakupdata function.",
        "PrestaShop is an open source e-commerce web application. Prior to version 8.1.1, SQL injection possible in the product search field, in BO's product page. Version 8.1.1 contains a patch for this issue. There are no known workarounds.",
        "webchess v1.0 was discovered to contain a SQL injection vulnerability via the $playerID parameter at mainmenu.php.",
        "social-media-skeleton is an uncompleted social media project. A SQL injection vulnerability in the project allows UNION based injections, which indirectly leads to remote code execution. Commit 3cabdd35c3d874608883c9eaf9bf69b2014d25c1 contains a fix for this issue.",
    ]

    predict(input_texts)


def predict(input_texts):
    # Load the trained model
    model_name = "./confidentiality_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_map = {"none": 0, "low": 1, "high": 2}

    # Tokenize the input texts
    tokenized_texts = tokenizer(
        input_texts, padding=True, truncation=True, return_tensors="pt", max_length=256
    )

    # Get logits from the model
    logits = model(
        tokenized_texts["input_ids"], attention_mask=tokenized_texts["attention_mask"]
    ).logits

    # Interpret the logits for each input text
    predicted_classes = torch.argmax(logits, dim=1)
    predicted_confidences = torch.softmax(logits, dim=1)

    predicted_labels = [
        list(label_map.keys())[list(label_map.values()).index(class_index.item())]
        for class_index in predicted_classes
    ]

    for i, input_text in enumerate(input_texts):
        print("Input Text:", input_text)
        print(
            "Predicted CVSS Confidentiality:",
            predicted_labels[i],
            f"- {predicted_confidences[i]}",
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
