import streamlit as st
import pandas as pd
import zipfile
import os
import io
import hashlib

# --- Helpers, Modules and Sandboxing ---
from helpers.normalize_text import normalize_text
from modules.init_session_state import init_session_state
from sandboxing.executa_codigo_seguro import executa_codigo_seguro

# ------- Agents -------
from agents.agente_limpeza_dados import agente_limpeza_dados
from agents.agente0 import agente0_clarifica_pergunta
from agents.agente1 import (
    agente1_identifica_arquivos,
    agente1_interpreta_contexto_arquivo,
    agente1_processa_arquivo_chunk
)
from agents.agente2 import agente2_gera_codigo_pandas_eda
from agents.agente3 import agente3_formatar_apresentacao

# --- RAG Components ---
from rag_components.create_faiss_index_for_chunk import create_faiss_index_for_chunk
from rag_components.retrieve_context import retrieve_context
from rag_components.save_progress import save_progress
from rag_components.load_progress import load_progress

# Importação da SentenceTransformer será feita via st.cache_resource

# --- Configurações Iniciais e Variáveis ---
# Streamlit Page Configuration - Foco em Light Mode (Claro, Amigável)
st.set_page_config(
    page_title="EDA com Gemini e RAG - Guiado por Passos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
CHUNK_SIZE = 1000

# --- Inicialização de Session State ---
init_session_state()

# --- Helpers para o Guia da Sidebar ---
def get_step_status(step_index):
    """Determina o status do passo para a sidebar."""
    if step_index == 1:
        return st.session_state.get('gemini_api_key') is not None
    elif step_index == 2:
        return st.session_state.get('zip_bytes') is not None and st.session_state.get('available_files')
    elif step_index == 3:
        return st.session_state.get('processed_percentage', 0) >= 100
    elif step_index == 4:
        return st.session_state.get('codigo_gerado') is not None
    return False

def render_step(step_index, description):
    """Renderiza um item do passo com um ícone de status."""
    is_done = get_step_status(step_index)
    icon = "✅" if is_done else "⏳"
    st.markdown(f"**{icon} Passo {step_index}:** {description}")

# --- Streamlit UI ---

# 1. CABEÇALHO E CHAVE API (Topo da Página Principal)
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.title("🗺️ Análise de Dados em Etapas")
    st.caption("Um fluxo de trabalho guiado para sua Análise Exploratória de Dados (EDA).")

with header_col2:
    with st.popover("🔑 API Key"):
        st.subheader("Configurar Gemini API Key")
        with st.form("api_key_form_guided"):
            api_key_input = st.text_input("Cole sua Chave:", value=st.session_state.get('gemini_api_key', ''), type="password", key="gemini_api_key_input_form_guided")
            submitted = st.form_submit_button("Salvar")
            if submitted:
                st.session_state['gemini_api_key'] = api_key_input
                st.success("Chave salva!")

st.markdown("---")

# --- Configuração da Sidebar (Foco no Guia e Conclusões) ---
with st.sidebar:
    st.header("📌 Guia de Progresso")
    
    # Guia Visual de 4 Passos
    render_step(1, "Chave API Configurada")
    render_step(2, "Arquivo Carregado e Contextualizado")
    render_step(3, "Dados Indexados (100% RAG)")
    render_step(4, "Primeira Análise Executada")
    
    st.markdown("---")
    
    st.header("📝 Histórico de Insights")
    if st.session_state.get('conclusoes_historico'):
        st.markdown(st.session_state['conclusoes_historico'])
    else:
        st.info("As conclusões da IA aparecerão aqui após a análise.")


# --- Seção 1: Upload e Preparação (Passos 2 e 3) ---
with st.container(border=True):
    st.header("1. Preparação do Arquivo")
    st.caption("Carregue o arquivo ZIP e inicie o processamento RAG.")
    
    col_upload, col_process = st.columns([1, 2])
    
    with col_upload:
        zipfile_input = st.file_uploader("📂 Upload do ZIP:", type=["zip"], label_visibility="collapsed")
        
        # Botão para listar arquivos
        if st.button("Listar Arquivos no ZIP", use_container_width=True, type="secondary"):
            if zipfile_input is not None:
                st.session_state['zip_bytes'] = zipfile_input.getvalue()
                st.session_state['zip_hash'] = hashlib.md5(st.session_state['zip_bytes']).hexdigest()
                
                # Reseta estados importantes
                st.session_state['selected_file_name'] = None 
                st.session_state['df'] = None 
                st.session_state['conclusoes_historico'] = "" 
                st.session_state['processed_percentage'] = 0
                st.session_state['available_files'] = []
                st.session_state['file_options_map'] = {}
                st.session_state['faiss_index'] = None
                st.session_state['documents'] = []

                with st.spinner("Analisando arquivos e gerando contexto inicial..."):
                    file_info_list = agente1_identifica_arquivos(st.session_state['zip_bytes'])
                    
                    if not file_info_list:
                        st.error("O ZIP não contém arquivos CSV, XLSX ou TXT válidos.")
                    else:
                        st.session_state['available_files'] = file_info_list
                        file_context_map = agente1_interpreta_contexto_arquivo(st.session_state.get('gemini_api_key'), file_info_list)
                        
                        options = {}
                        for info in file_info_list:
                            context = file_context_map.get(info['name'], "Contexto não gerado.")
                            options[info['name']] = f"**{info['name']}** - {context}"
                        
                        st.session_state['file_options_map'] = options
                        st.success(f"Encontrados {len(file_info_list)} arquivos. Prossiga para a seleção.")
            else:
                 st.warning("Por favor, carregue um arquivo ZIP primeiro.")

    with col_process:
        if st.session_state['available_files']:
            st.subheader("Selecione o Conjunto de Dados")
            
            options_list = list(st.session_state['file_options_map'].keys())
            display_options = [st.session_state['file_options_map'][key] for key in options_list]
            
            default_index = 0
            try:
                if st.session_state['selected_file_name'] in options_list:
                    default_index = options_list.index(st.session_state['selected_file_name'])
            except:
                 default_index = 0

            selected_display = st.selectbox(
                "Arquivo para Indexação:",
                options=display_options,
                index=default_index,
                key="file_selection_select_guided",
                format_func=lambda x: x.split(" - ")[0]
            )
            
            selected_file_name = options_list[display_options.index(selected_display)]
            st.session_state['selected_file_name'] = selected_file_name
            
            selected_file_info = next((info for info in st.session_state['available_files'] if info["name"] == selected_file_name), None)
            
            # Exibe contexto do arquivo
            st.markdown(f"**Contexto:** {st.session_state['file_options_map'].get(st.session_state['selected_file_name'], 'Nenhum contexto.')}")
            
            # Botão de Análise (Indexação RAG)
            if st.button(f"🚀 Iniciar Processamento RAG: {selected_file_name}", use_container_width=True, type="primary") and selected_file_info:
                
                # --- INÍCIO DA LÓGICA DE CARREGAMENTO/CHUNKED (MANTIDA) ---
                expected_num_cols = selected_file_info['num_cols']
                st.info(f"Tentando carregar progresso anterior para **{selected_file_name}**...")
                
                # Tenta carregar o progresso anterior
                df_loaded, index_loaded, docs_loaded, lines_loaded_processed = load_progress(st.session_state['zip_hash'], selected_file_name)
                
                st.session_state['df'] = df_loaded
                st.session_state['faiss_index'] = index_loaded
                st.session_state['documents'] = docs_loaded
                st.session_state['current_chunk_start'] = lines_loaded_processed 
                
                # Tenta obter o total de linhas real do arquivo
                total_lines_file = 0
                try:
                    ext = selected_file_info['extension']
                    if ext == '.csv' or ext == '.txt':
                        with zipfile.ZipFile(io.BytesIO(st.session_state['zip_bytes']), "r") as z:
                            with z.open(selected_file_name, 'r') as file_in_zip:
                                total_lines_file = sum(1 for line in io.TextIOWrapper(file_in_zip, encoding='utf-8', errors='ignore')) - 1 
                    else:
                        total_lines_file = CHUNK_SIZE * 50 
                        
                except Exception as e:
                    total_lines_file = CHUNK_SIZE * 10 
                
                st.session_state['total_lines'] = max(total_lines_file, lines_loaded_processed)
                st.session_state['file_name_context'] = normalize_text(os.path.splitext(selected_file_name)[0].upper().replace('_', ' ').replace('-', ' '))

                # Verifica se o carregamento foi completo ou se precisa continuar
                if lines_loaded_processed > 0 and lines_loaded_processed >= total_lines_file:
                    st.session_state['df'] = agente_limpeza_dados(st.session_state['df'])
                    st.session_state['processed_percentage'] = 100
                    st.success(f"Processamento concluído ({len(st.session_state['df'])} linhas).")
                    st.progress(1.0, text="Processamento 100% concluído.")
                    st.rerun() 
                
                # Se o carregamento parcial ocorreu, precisamos continuar
                elif lines_loaded_processed > 0 and lines_loaded_processed < total_lines_file:
                    st.info(f"Progresso parcial encontrado ({lines_loaded_processed} linhas). Continuaremos...")
                    st.session_state['df_columns'] = st.session_state['df'].columns 
                    st.session_state['df'] = agente_limpeza_dados(st.session_state['df']) 
                
                # --- INÍCIO DO NOVO PROCESSAMENTO ---
                else:
                    st.info(f"Iniciando novo processamento para **{selected_file_name}** ({st.session_state['total_lines']} linhas estimadas)...")
                    st.session_state['df'] = None
                    st.session_state['faiss_index'] = None
                    st.session_state['documents'] = []
                    st.session_state['conclusoes_historico'] = ""
                    st.session_state['df_columns'] = None
                    st.session_state['processed_percentage'] = 0
                    st.session_state['cleaned_status'] = {}
                    st.session_state['current_chunk_start'] = 0
                    lines_loaded_processed = 0


                # Loop de processamento de chunks
                progress_bar = st.progress(lines_loaded_processed / st.session_state['total_lines'], 
                                           text=f"Criando embeddings e índice RAG... {lines_loaded_processed}/{st.session_state['total_lines']} linhas...")
                
                start_row = lines_loaded_processed
                
                while start_row < st.session_state['total_lines'] or start_row == 0:
                    
                    chunk_processed, msg = agente1_processa_arquivo_chunk(
                        st.session_state['zip_bytes'], 
                        selected_file_name, 
                        start_row, 
                        CHUNK_SIZE, 
                        st.session_state['df_columns'],
                        expected_num_cols
                    )
                    
                    if chunk_processed is not None:
                        chunk_processed = agente_limpeza_dados(chunk_processed)
                        
                        if st.session_state['df'] is None:
                            st.session_state['df'] = chunk_processed
                            st.session_state['df_columns'] = chunk_processed.columns
                            expected_num_cols = len(st.session_state['df_columns'])
                        else:
                            if len(chunk_processed.columns) == len(st.session_state['df_columns']):
                                chunk_processed.columns = st.session_state['df_columns']
                            st.session_state['df'] = pd.concat([st.session_state['df'], chunk_processed], ignore_index=True)
                        
                        create_faiss_index_for_chunk(chunk_processed)
                        
                        start_row += len(chunk_processed)
                        st.session_state['current_chunk_start'] = start_row
                        
                        if len(chunk_processed) < CHUNK_SIZE and start_row < st.session_state['total_lines']:
                             st.session_state['total_lines'] = start_row
                             
                        progress_value = min(start_row / st.session_state['total_lines'], 1.0) if st.session_state['total_lines'] > 0 else 1.0
                        st.session_state['processed_percentage'] = progress_value * 100
                        
                        progress_bar.progress(progress_value, 
                                              text=f"Criando embeddings e índice RAG... {start_row}/{st.session_state['total_lines']} linhas - {st.session_state['processed_percentage']:.1f}%")
                        
                        save_progress(st.session_state['zip_hash'], st.session_state['df'], st.session_state['faiss_index'], st.session_state['documents'], st.session_state['total_lines'])
                        
                        if len(chunk_processed) < CHUNK_SIZE:
                            st.session_state['total_lines'] = start_row 
                            break
                            
                    else:
                        if "todos os lotes concluído" in msg:
                            st.session_state['total_lines'] = start_row 
                            break
                        st.error(msg)
                        break
                
                if st.session_state['df'] is not None and len(st.session_state['df']) > 0:
                    st.session_state['total_lines'] = len(st.session_state['df'])
                    st.success(f"Processamento de **{selected_file_name}** concluído! Total de linhas: {len(st.session_state['df'])}")
                    progress_bar.progress(1.0, text="Processamento 100% concluído. A ferramenta está pronta para a análise.")
                    st.session_state['processed_percentage'] = 100
                    st.rerun() 
                else:
                    progress_bar.empty()
                    st.error("Falha ao carregar o arquivo. Verifique se o formato está correto.")


st.markdown("---")

# --- Seção 2: Consulta e Resultados (Passo 4) ---
with st.container(border=True):
    st.header("2. Análise e Geração de Insights")

    if st.session_state['df'] is None or st.session_state['processed_percentage'] < 100:
        st.warning("⚠️ Aguardando a conclusão do **Processamento RAG** (100% na seção acima).")
    else:
        st.success(f"Dados do arquivo **{st.session_state['selected_file_name']}** prontos para análise.")

        # Área de Consulta
        pergunta = st.text_area(
            "💬 O que você gostaria de analisar?",
            value=st.session_state.get('user_query_input_widget', ""),
            placeholder="Ex: Qual a média de 'Preço' por 'Categoria'? Exiba os 10 maiores clientes.",
            height=80,
            key="user_query_input_widget_guided"
        )
        
        col_btn, col_resumo = st.columns([1, 2])
        
        with col_btn:
            if st.button("🔎 Consultar IA", type="primary", use_container_width=True):
                st.session_state['consultar_ia'] = True
        
        with col_resumo:
            st.metric(
                label="Status do Processamento",
                value=f"{st.session_state['processed_percentage']:.1f}%",
                delta=f"{len(st.session_state['df'])} linhas"
            )

        st.markdown("---")

        # --- Lógica de Execução da Consulta ---
        if 'consultar_ia' in st.session_state and st.session_state['consultar_ia']:
            st.session_state['consultar_ia'] = False
            
            if not st.session_state.get('gemini_api_key'):
                st.error("Por favor, insira e salve sua API Key do Gemini no popover do topo.")
            elif st.session_state['faiss_index'] is None or st.session_state['faiss_index'].ntotal == 0:
                st.warning("O índice RAG não foi criado. Clique em 'Iniciar Processamento RAG'.")
            else:
                pergunta_original = pergunta 
                
                # --- ETAPA DE CLARIFICAÇÃO ---
                with st.spinner("Clarificando sua pergunta..."):
                    pergunta_clarificada = agente0_clarifica_pergunta(pergunta_original, st.session_state['gemini_api_key'])
                
                pergunta_para_ia = pergunta_clarificada 
                
                if pergunta_para_ia != pergunta_original:
                     st.warning(f"Sua consulta foi clarificada para: **{pergunta_para_ia}**")
                
                with st.spinner("Gerando código e analisando dados..."):
                    df_to_use = st.session_state['df']
                    faiss_index = st.session_state['faiss_index']
                    documents = st.session_state['documents']
                    api_key = st.session_state['gemini_api_key']

                    # 1. Recupera o Contexto (RAG)
                    retrieved_context = retrieve_context(pergunta_para_ia, faiss_index, documents)
                    
                    # 2. Gera Código e Conclusão
                    codigo_gerado, conclusoes = agente2_gera_codigo_pandas_eda(
                        pergunta_para_ia, 
                        api_key, 
                        df_to_use, 
                        retrieved_context, 
                        st.session_state['conclusoes_historico'],
                        st.session_state['file_name_context'] 
                    )
                    
                    if conclusoes:
                        st.session_state['conclusoes_historico'] += f"\n- {conclusoes}"
                    
                    st.session_state['codigo_gerado'] = codigo_gerado
                    
                    # 3. Executa o Código
                    if codigo_gerado.startswith("Erro:"):
                        st.error(codigo_gerado)
                    else:
                        resultado_texto, resultado_df, erro_execucao, img_bytes = executa_codigo_seguro(codigo_gerado, df_to_use)
                        
                        st.session_state['resultado_texto'] = resultado_texto
                        st.session_state['resultado_df'] = resultado_df
                        st.session_state['erro_execucao'] = erro_execucao
                        st.session_state['img_bytes'] = img_bytes
                        
                        if erro_execucao:
                            st.error(f"❌ Erro na Execução do Código:\n{erro_execucao}")
                        
                        st.rerun() 

        # --- Seção de Resultados (Abas para organização do Output) ---
        if st.session_state.get('codigo_gerado'):
            
            st.subheader("Resultados da Análise")
            
            # Usando Abas para organizar os resultados (Tabela, Gráfico, Código, PDF)
            tab_data, tab_plot, tab_code, tab_pdf = st.tabs(["📊 Dados/Texto", "📈 Gráfico", "💻 Código", "📄 Relatório"])

            # ABA 1: Resultado (Tabela/Texto)
            with tab_data:
                if st.session_state.get('resultado_df') is not None and not st.session_state['resultado_df'].empty:
                    resultado_df = st.session_state['resultado_df']
                    
                    if 'INFORMAÇÃO' in resultado_df.columns:
                        st.markdown("##### Informação Detalhada (Texto Completo):")
                        table_markdown = "| | INFORMAÇÃO |\n"
                        table_markdown += "| :--- | :--- |\n"
                        for index, row in resultado_df.iterrows():
                            index_display = index if resultado_df.index.name is None else row.name
                            table_markdown += f"| **{index_display}** | {row['INFORMAÇÃO']} |\n"
                        st.markdown(table_markdown)
                    else:
                        st.dataframe(resultado_df, use_container_width=True)
                else:
                    st.info("Nenhum DataFrame gerado.")
                    
                if st.session_state.get('resultado_texto'):
                    st.markdown("---")
                    st.markdown("##### Mensagem da IA:")
                    st.write(st.session_state['resultado_texto'])


            # ABA 2: Gráfico
            with tab_plot:
                if st.session_state.get('img_bytes'):
                    st.image(st.session_state['img_bytes'], caption="Gráfico da Análise", use_container_width=True)
                else:
                    st.warning("Nenhum gráfico gerado na última consulta.")

            # ABA 3: Código Python
            with tab_code:
                st.code(st.session_state['codigo_gerado'], language='python')

            # ABA 4: Relatório PDF
            with tab_pdf:
                if st.session_state.get('codigo_gerado'):
                    if st.button("Gerar e Baixar Relatório PDF", key="btn_gerar_pdf_guided", type="secondary"):
                        with st.spinner("Preparando PDF..."):
                            pdf_bytes = None
                            try:
                                _, _, pdf_bytes = agente3_formatar_apresentacao(
                                    st.session_state.get('resultado_texto', ""), 
                                    st.session_state.get('resultado_df'), 
                                    pergunta, 
                                    st.session_state.get('img_bytes')
                                )
                            except Exception as e:
                                st.error(f"Erro ao gerar PDF: {e}")
                                
                            if pdf_bytes:
                                st.download_button(
                                    label="⬇️ Baixar PDF",
                                    data=pdf_bytes,
                                    file_name="relatorio_eda_guiado.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.warning("Não foi possível gerar o PDF.")
                else:
                    st.warning("Execute uma consulta antes.")

st.markdown("""
<hr style="border: 0.5px solid #333; margin-top: 3em; margin-bottom: 1em;">

<div style="text-align:center; color: #888; font-size: 14px;">
    Desenvolvido com 💡 por 
    <b style="color:#4CAF50;">Erlon Lopes Dias</b> 
    — Grupo <b style="color:#00BFFF;">TenkAI</b> ⚡<br>
    <span style="font-size:12px;">© 2025 — Projeto I2A2 | EDA Agent</span>
</div>
""", unsafe_allow_html=True)
