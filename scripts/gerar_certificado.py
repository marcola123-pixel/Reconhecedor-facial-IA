"""
Gera um certificado TLS autoassinado (localhost / 127.0.0.1), válido por
365 dias, para o servidor rodar em HTTPS numa porta segura.

Uso:
    pip install cryptography   # já está no requirements.txt
    python gerar_certificado.py

Gera certs/key.pem (chave privada) e certs/cert.pem (certificado).
"""

import datetime
import ipaddress
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

PASTA = "certs"


def main():
    os.makedirs(PASTA, exist_ok=True)

    chave = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    nome = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Porteiro Eletronico - Projeto UFMS"),
    ])

    agora = datetime.datetime.now(datetime.timezone.utc)

    certificado = (
        x509.CertificateBuilder()
        .subject_name(nome)
        .issuer_name(nome)
        .public_key(chave.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(agora)
        .not_valid_after(agora + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(chave, hashes.SHA256())
    )

    caminho_key = os.path.join(PASTA, "key.pem")
    caminho_cert = os.path.join(PASTA, "cert.pem")

    with open(caminho_key, "wb") as f:
        f.write(chave.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    with open(caminho_cert, "wb") as f:
        f.write(certificado.public_bytes(serialization.Encoding.PEM))

    print(f"Certificado gerado:\n  {caminho_cert}\n  {caminho_key}\n(validade: 365 dias)")


if __name__ == "__main__":
    main()
