from Config_Loader import load_config
import os
import torch
from Quadrakill_Processor import processor
from Model.QuadraKill import QuadraKill
import torch.optim as optim
import torch.nn.functional as F

def predict(domain):
    config = load_config()

    flag, quadrakill = processor(domain, config)
    if flag:
        try:
            model_repo = config["model_repo"]
            model_path = os.path.join(model_repo, config["quadrakill"])

            checkpoint = torch.load(model_path, map_location= "cpu")

            # Kiểm tra checkpoint hợp lệ
            if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint:
                print(f"Lỗi: Không tìm thấy 'model_state_dict' hoặc 'optimizer_state_dict' trong checkpoint.")
                return

            model = QuadraKill(1536, 53)
            model.load_state_dict(checkpoint['model_state_dict'])

            optimizer = optim.Adam(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.eval()

            # Lấy dữ liệu từ dict trả về
            domain_emb = quadrakill['domain_emb']
            headers_emb = quadrakill['headers_emb']
            content_emb = quadrakill['content_emb']
            normal_feat = quadrakill['features']

            with torch.no_grad():
                outputs = model(domain_emb, headers_emb, content_emb, normal_feat)

                softmax_output = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(softmax_output, dim=1)
                predicted_prob = softmax_output[0][predicted_class].item()

                label = 'black' if predicted_class.item() == 1 else 'clean'
                print(f"Kết quả {domain}: Lớp '{label}' với xác suất: {predicted_prob:.4f}")
        except Exception as e:
            print(f"Lỗi khi xử lý domain {domain}: {e}")
    else:
        print("Sẽ load model XGBoost hoặc xử lý khác")

print("lấy bừa trên mạng")
predict("gayteam.club")
predict("xemtrai.top")
predict("yandex.ru")
predict("dzen.ru")
predict("suckhoedoisong.vn")
predict("nytimes.com")
predict("twitter.com")
predict("cnn.com")
predict("hackerrank.com")
predict("hust.edu.vn")
predict("neu.edu.vn")
predict("daihoc.fpt.edu.vn")
predict("vnexpress.net")
predict("vietnamnet.vn")
predict("mard.gov.vn")
predict("hoanmy.com")
predict("thaithuonghoang.vn")
predict("minhlongbook.vn")
predict("www.fahasa.com")
print("clean trong train")
predict("3nod.com.cn")
predict("breathingspacelondon.org.uk")
predict("fieldhockey.ca")
predict("nfocus.co.uk")
predict("research.njau.edu.cn")
predict("suc.cczu.edu.cn")
predict("twigtwisters.co.uk")
predict("greenacademy.edu.vn")
predict("hailam.com.vn")
predict("haiphong.gov.vn")
predict("myad.vn")
predict("tamdao.vinhphuc.gov.vn")
print("black trong train")
predict("1livecam.com")
predict("sextoyslima.com")
predict("sextreffenx.de")
predict("www.kanikaji.com")
predict("jav-dl.com")
predict("istockbargains.com")
predict("camania.club")
predict("0porn.info")
