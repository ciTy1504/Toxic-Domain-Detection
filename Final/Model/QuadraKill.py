import torch
from torch import nn

class QuadraKill(nn.Module):
    def __init__(self, input_dim, normal_dim, domain_dim=128, domain_header_dim=256, domain_content_dim=384, normal_dim_out=384):
        super(QuadraKill, self).__init__()

        # Định nghĩa các lớp fully connected (FC) cho domain, header, content và normal
        self.FC_domain = nn.Sequential(
            nn.Linear(input_dim, domain_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.FC_domain_header = nn.Sequential(
            nn.Linear(input_dim, domain_header_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.FC_domain_content = nn.Sequential(
            nn.Linear(input_dim, domain_content_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.FC_normal = nn.Sequential(
            nn.Linear(normal_dim, normal_dim_out),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        total_input_dim = domain_dim + domain_header_dim + domain_content_dim + normal_dim_out

        # Định nghĩa phần classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)  # Output cho 2 lớp: classification (ví dụ: phân loại có/không)
        )

    def forward(self, domain, header, content, normal_features):
        # Tiến hành forward qua các lớp đã định nghĩa
        d = self.FC_domain(domain)
        d_h = self.FC_domain_header(header)
        d_c = self.FC_domain_content(content)
        n = self.FC_normal(normal_features)

        # Kết hợp các features lại
        x = torch.cat([d, d_h, d_c, n], dim=1)

        # Phân loại đầu ra
        out = self.classifier(x)
        return out