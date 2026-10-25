// A decisão liberado/negado vem do SERVIDOR (limiar validado na Parte 1); o navegador só exibe.
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const btnLigar = document.getElementById('btnLigar');
const dica = document.getElementById('dica');
const statusBadge = document.getElementById('statusBadge');
const estadoTexto = document.getElementById('estadoTexto');
const pessoaTexto = document.getElementById('pessoaTexto');
const confiancaTexto = document.getElementById('confiancaTexto');
const barraConfianca = document.getElementById('barraConfianca');
const limiarValor = document.getElementById('limiarValor');
const logLista = document.getElementById('logLista');
const relogio = document.getElementById('relogio');

let rodando = false;
let intervalo = null;
let emAndamento = false;
let ultimoRegistro = { nome: null, liberado: null };

setInterval(() => { relogio.textContent = new Date().toLocaleTimeString('pt-BR'); }, 1000);

btnLigar.addEventListener('click', async () => {
  if (!rodando) {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
      rodando = true;
      btnLigar.textContent = 'Desligar portaria';
      btnLigar.classList.add('ligado');
      dica.textContent = 'Portaria ativa';
      intervalo = setInterval(capturarEAnalisar, 800);
    } catch (e) {
      dica.textContent = 'Não consegui acessar a câmera: ' + e.message;
    }
  } else {
    rodando = false;
    clearInterval(intervalo);
    if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
    btnLigar.textContent = 'Ligar portaria';
    btnLigar.classList.remove('ligado');
    dica.textContent = 'Portaria desligada';
    definirEstado('aguardando', 'Aguardando', '—', 'similaridade: —', 0);
  }
});

function definirEstado(classe, estado, pessoa, detalhe, pct) {
  statusBadge.className = 'card status-badge ' + classe;
  estadoTexto.textContent = estado;
  pessoaTexto.textContent = pessoa;
  confiancaTexto.textContent = detalhe;
  barraConfianca.style.width = pct + '%';
  barraConfianca.style.background =
    classe === 'liberado' ? 'var(--ok)' : classe === 'negado' ? 'var(--deny)' : 'var(--idle)';
}

async function capturarEAnalisar() {
  if (!rodando || emAndamento || video.readyState < 2) return;
  emAndamento = true;                      // evita acumular requisições se o servidor demorar
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  try {
    const resp = await fetch('/reconhecer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.8) }),
    });
    if (!resp.ok) { dica.textContent = 'Servidor respondeu ' + resp.status; return; }
    processarResultado(await resp.json());
  } catch (e) {
    dica.textContent = 'Erro ao falar com o servidor: ' + e.message;
  } finally {
    emAndamento = false;
  }
}

function processarResultado(r) {
  if (!r.rosto_detectado) {
    definirEstado('aguardando', 'Nenhum rosto', '—', 'similaridade: —', 0);
    return;
  }
  const pct = Math.max(0, Math.round(r.similaridade * 100));
  if (r.liberado) {
    definirEstado('liberado', 'Acesso liberado', r.nome, 'similaridade: ' + pct + '%', pct);
  } else {
    definirEstado('negado', 'Acesso negado', 'Desconhecido', 'similaridade: ' + pct + '%', pct);
  }
  registrarLog(r.liberado ? r.nome : 'desconhecido', r.similaridade, r.liberado);
}

function registrarLog(nome, similaridade, liberado) {
  if (ultimoRegistro.nome === nome && ultimoRegistro.liberado === liberado) return;
  ultimoRegistro = { nome, liberado };
  const vazio = logLista.querySelector('.vazio');
  if (vazio) vazio.remove();

  const item = document.createElement('div');
  item.className = 'log-item ' + (liberado ? 'liberado' : 'negado');
  const n = document.createElement('span');
  n.className = 'nome';
  n.textContent = nome;                    // textContent: nome nunca é interpretado como HTML
  const m = document.createElement('span');
  m.className = 'meta mono';
  m.textContent = Math.round(similaridade * 100) + '% · ' + new Date().toLocaleTimeString('pt-BR');
  item.append(n, m);
  logLista.prepend(item);
  while (logLista.children.length > 10) logLista.removeChild(logLista.lastChild);
}

fetch('/status').then(r => r.json()).then(s => {
  limiarValor.textContent = s.limiar.toFixed(2);
  dica.textContent = s.pessoas_cadastradas === 0
    ? 'Nenhuma pessoa cadastrada — use scripts/enroll.py antes de testar.'
    : 'Clique em "Ligar portaria" para começar';
});
