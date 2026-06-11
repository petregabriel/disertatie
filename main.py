import pandas as pd
import numpy as np
import math
import scipy.stats as stats

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox
import sys

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.widgets.scrolled import ScrolledText
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel

# ==========================================
# CLASĂ PENTRU REDIRECȚIONAREA CONSOLEI ÎN GUI
# ==========================================
class PrintRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Auto-scroll la final

    def flush(self):
        pass


# ==========================================
# LOGICA DE BAZĂ & BACKEND IPOTEZE
# ==========================================

# =======================================================
# IPOTEZA 1: CORELAȚII SPEARMAN
# =======================================================
def analizeaza_corelatii_spearman(file_path):
    try:
        df = pd.read_excel(file_path)

        col_calitate = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Calitatea și relevanța conținutului postat (text, imagini, video)]'
        col_claritate = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Claritatea informațiilor oferite]'
        col_satisfactie = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Conținutul social media îmi îmbunătățește satisfacția față de un brand.]'

        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}

        df['scor_calitate'] = df[col_calitate].map(influence_mapping)
        df['scor_claritate'] = df[col_claritate].map(influence_mapping)
        df['scor_satisfactie'] = df[col_satisfactie].map(agreement_mapping)

        print("=" * 90)
        print("IPOTEZA 1: CORELAȚII SPEARMAN (Calitate & Claritate vs. Satisfacție)")
        print("=" * 90)

        df_corr1 = df[['scor_calitate', 'scor_satisfactie']].dropna()
        rho1, p_val1 = stats.spearmanr(df_corr1['scor_calitate'], df_corr1['scor_satisfactie'])
        print("1. [Calitatea conținutului] vs [Satisfacția față de brand]")
        print(
            f"   -> Rho = {rho1:.4f} | p = {p_val1:.2e} | {' Semnificativ' if p_val1 < 0.05 else ' Nesemnificativ'}\n")

        df_corr2 = df[['scor_claritate', 'scor_satisfactie']].dropna()
        rho2, p_val2 = stats.spearmanr(df_corr2['scor_claritate'], df_corr2['scor_satisfactie'])
        print("2. [Claritatea informațiilor] vs [Satisfacția față de brand]")
        print(
            f"   -> Rho = {rho2:.4f} | p = {p_val2:.2e} | {' Semnificativ' if p_val2 < 0.05 else ' Nesemnificativ'}")

        print("=" * 90)

    except Exception as e:
        print(f"Eroare la calculul corelațiilor Spearman: {e}")


# =======================================================
# IPOTEZA 2: ANALIZA RECENZIILOR ȘI CONȚINUTULUI + T-TEST
# =======================================================
def analiza_ipoteza_2_recenzii(file_path):
    try:
        df = pd.read_excel(file_path)
        col_recenzii = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Recenziile și mărturiile altor clienți, vizibile pe pagină]'
        col_acord = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Recenziile și comentariile altor clienți îmi influențează decizia de cumpărare.]'
        col_continut = 'Ce tip de conținut vă influențează cel mai mult?'

        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}

        target_phrases = ['recenzii ale altor clienți', 'videoclipuri în format lung', 'videoclipuri scurte',
                          'conținut creativ']

        def contine_macar_unul(text):
            if pd.isna(text): return 0
            text_lower = str(text).lower()
            for phrase in target_phrases:
                if phrase in text_lower: return 1
            return 0

        df['flag_continut'] = df[col_continut].apply(contine_macar_unul)
        total_resp = len(df)
        au_bifat = df['flag_continut'].sum()

        procentaj_continut = (au_bifat / total_resp) * 100 if total_resp > 0 else 0
        conditie_1 = procentaj_continut > 50

        media_importanta_recenzii = df[col_recenzii].map(influence_mapping).dropna().mean()
        conditie_2 = media_importanta_recenzii > 3

        media_acord_recenzii = df[col_acord].map(agreement_mapping).dropna().mean()
        conditie_3 = media_acord_recenzii > 3

        print("=" * 90)
        print("IPOTEZA 2: ANALIZA RECENZIILOR ȘI A TIPURILOR DE CONȚINUT")
        print("=" * 90)

        print(f"1. Importanța Recenziilor în decizia de cumpărare (Scară 1-5):")
        print(f"   • Media obținută: {media_importanta_recenzii:.2f}")
        print(f"   -> Validare (Media > 3): {' DA' if conditie_2 else ' NU'}\n")

        print(f"2. Acord: 'Recenziile îmi influențează decizia' (Scară 1-5):")
        print(f"   • Media obținută: {media_acord_recenzii:.2f}")
        print(f"   -> Validare (Media > 3): {' DA' if conditie_3 else ' NU'}\n")

        print(f"3. Proporția respondenților care consumă Recenzii, Video sau Conținut Creativ:")
        print(f"   • Rezultat: {au_bifat} din {total_resp} respondenți ({procentaj_continut:.1f}%)")
        print(f"   -> Validare (Peste 50%): {' DA' if conditie_1 else ' NU'}")

        print("-" * 90)
        if conditie_1 and conditie_2 and conditie_3:
            print("CONCLUZIE IPOTEZA 2: VALIDATĂ!")
        else:
            print("CONCLUZIE IPOTEZA 2: VALIDARE PARȚIALĂ / NEVALIDATĂ")
        print("=" * 90)

    except Exception as e:
        print(f"A apărut o eroare la analiza Ipotezei 2: {e}")


def analiza_t_test_influenta_negativa(file_path):
    try:
        df = pd.read_excel(file_path)
        col_da_nu = 'Ați fost vreodată influențat să NU cumpărați un produs sau serviciu din cauza imaginii slabe sau lipsei de prezență a unei afaceri pe social media?'
        col_cred = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Credibilitatea brandului pe social media]'
        col_prez_incredere = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Prezența pe social media îmi arată că un brand este de încredere.]'

        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}

        df['scor_cred'] = df[col_cred].map(influence_mapping)
        df['scor_prez'] = df[col_prez_incredere].map(agreement_mapping)
        df['medie_incredere'] = df[['scor_cred', 'scor_prez']].mean(axis=1)

        df_da = df[df[col_da_nu].astype(str).str.startswith('Da')]
        df_nu = df[df[col_da_nu].astype(str).str.startswith('Nu,')]

        raspunsuri_da = df_da['medie_incredere'].dropna()
        raspunsuri_nu = df_nu['medie_incredere'].dropna()

        t_stat, p_value = stats.ttest_ind(raspunsuri_da, raspunsuri_nu, equal_var=True)

        print("=" * 90)
        print("IPOTEZA 2: TESTUL T-STUDENT (Influența Negativă a Imaginii Brandului)")
        print("=" * 90)
        print(f"  -> Valoarea p = {p_value:.4f}")
        if p_value < 0.05:
            print("  -> CONCLUZIE:  Diferență semnificativă statistic (p < 0.05).")
        else:
            print("  -> CONCLUZIE:  Fără diferență semnificativă (p >= 0.05).")
        print("=" * 90)

    except Exception as e:
        print(f"Eroare T-Test: {e}")


# =======================================================
# IPOTEZA 3: REGRESIE LINIARĂ MULTIPLĂ
# =======================================================
from statsmodels.miscmodels.ordinal_model import OrderedModel


def analizeaza_regresie_ordinala(file_path):
    try:
        df = pd.read_excel(file_path)

        # Mapping-uri
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}
        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}

        # Funcție de transformare Da/Nu în 1/0
        def binary_map(x):
            return 1 if str(x).lower().startswith('da') else 0

        df_reg = pd.DataFrame({
            'Y': df[
                'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Răspunsurile rapide pe social media îmi cresc satisfacția față de o companie.]'].map(
                agreement_mapping),
            'Timp': df[
                'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Timpul de răspuns la mesaje/comentarii]'].map(
                influence_mapping),
            'Prof': df[
                'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Profesionalismul comunicării]'].map(
                influence_mapping),
            # Noile variabile
            'Contact_Tehnic': df[
                'Ați contactat vreodată o afacere prin social media pentru suport tehnic sau întrebări legate de produse?'].apply(
                binary_map),
            'Contact_Info': df[
                'Ați contactat vreodată o afacere prin social media pentru informații, suport sau reclamații?'].apply(
                binary_map)
        }).dropna()

        X = df_reg[['Timp', 'Prof', 'Contact_Tehnic', 'Contact_Info']]
        Y = df_reg['Y']

        mod = OrderedModel(Y, X, distr='logit')
        res = mod.fit(method='bfgs')

        print("=" * 90)
        print("REZULTATE REGRESIE ORDINALĂ EXTINSĂ (Cu variabile de contact)")
        print("=" * 90)
        print(res.summary())
        print("=" * 90)

    except Exception as e:
        print(f"Eroare la calculul regresiei ordinale: {e}")

def analizeaza_regresie_liniara_multipla(file_path):
    try:
        df = pd.read_excel(file_path)

        # Mapping-uri
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}
        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}

        # Variabila dependentă (Y)
        y_col = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Răspunsurile rapide pe social media îmi cresc satisfacția față de o companie.]'
        Y = df[y_col].map(agreement_mapping)

        # Variabile independente (X)
        x_cols = [
            'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Timpul de răspuns la mesaje/comentarii]',
            'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Profesionalismul comunicării]'
        ]

        X = pd.DataFrame()
        X['Timp'] = df[x_cols[0]].map(influence_mapping)
        X['Profesionalism'] = df[x_cols[1]].map(influence_mapping)

        # Eliminăm rândurile cu date lipsă
        data = pd.concat([Y, X], axis=1).dropna()
        Y = data[y_col]
        X = data[['Timp', 'Profesionalism']]
        X = sm.add_constant(X)  # Adăugăm constanta (intercept)

        # 1. Modelul OLS
        model = sm.OLS(Y, X).fit()

        # 2. VIF (Multicoliniaritate)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

        print("=" * 90)
        print("REZULTATE REGRESIE LINIARĂ MULTIPLĂ")
        print("=" * 90)
        print(model.summary())

        print("\n🔍 VERIFICARE MULTICOLINIARITATE (VIF):")
        print(vif_data)
        print("(VIF > 5 indică probleme de multicoliniaritate)")

        # 3. Analiza Reziduurilor (Grafic)
        plt.figure(figsize=(8, 5))
        plt.scatter(model.fittedvalues, model.resid)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valori predictate')
        plt.ylabel('Reziduurile')
        plt.title('Analiza Reziduurilor (Homoscedasticitate)')
        plt.show()

    except Exception as e:
        print(f"Eroare regresie: {e}")


# =======================================================
# IPOTEZA 4: ANALIZA PROMOȚIILOR (GRUP 'DA' vs GRUP 'NU' - MEDIANĂ)
# =======================================================
def analiza_ipoteza_4_promotii(file_path):
    try:
        df = pd.read_excel(file_path)

        col_oferte_imp = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Ofertele speciale și promoțiile anunțate exclusiv pe social media]'
        col_oferte_acord = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Promoțiile comunicate prin canale sociale mă determină să cumpăr.]'
        col_continut = 'Ce tip de conținut vă influențează cel mai mult?'

        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}

        # Împărțim respondenții în 2 grupuri complet separate
        df['flag_promotii'] = df[col_continut].astype(str).str.contains('Promoții și oferte').astype(int)
        grup_da = df[df['flag_promotii'] == 1]
        grup_nu = df[df['flag_promotii'] == 0]

        # Extragem scorurile
        scoruri_imp_da = grup_da[col_oferte_imp].map(influence_mapping).dropna()
        scoruri_acord_da = grup_da[col_oferte_acord].map(agreement_mapping).dropna()

        scoruri_imp_nu = grup_nu[col_oferte_imp].map(influence_mapping).dropna()
        scoruri_acord_nu = grup_nu[col_oferte_acord].map(agreement_mapping).dropna()

        # Calculăm medianele
        mediana_imp_da = scoruri_imp_da.median() if not scoruri_imp_da.empty else 0
        mediana_acord_da = scoruri_acord_da.median() if not scoruri_acord_da.empty else 0

        mediana_imp_nu = scoruri_imp_nu.median() if not scoruri_imp_nu.empty else 0
        mediana_acord_nu = scoruri_acord_nu.median() if not scoruri_acord_nu.empty else 0

        # Validarea ipotezei (Grup Promoții >= Grup Fără Promoții)
        validare_imp = mediana_imp_da >= mediana_imp_nu
        validare_acord = mediana_acord_da >= mediana_acord_nu

        print("=" * 90)
        print("IPOTEZA 4: ANALIZA IMPACTULUI PROMOȚIILOR (GRUP 'DA' vs GRUP 'NU') - CALCUL MEDIANĂ")
        print("=" * 90)
        print(f"Număr persoane care AU ALES promoții: {len(grup_da)}")
        print(f"Număr persoane care NU AU ALES promoții: {len(grup_nu)}\n")

        print("1. Cât de importante sunt ofertele speciale în decizia de cumpărare? (Scară 1-5)")
        print(f"   • Mediana Grupului FĂRĂ Promoții: {mediana_imp_nu}")
        print(f"   • Mediana Grupului CU Promoții: {mediana_imp_da}")
        print(f"   -> Validare (Cu Promoții >= Fără Promoții): {' VALIDAT' if validare_imp else ' NEVALIDAT'}\n")

        print("2. 'Promoțiile comunicate pe social media mă determină să cumpăr' (Acord 1-5)")
        print(f"   • Mediana Grupului FĂRĂ Promoții: {mediana_acord_nu}")
        print(f"   • Mediana Grupului CU Promoții: {mediana_acord_da}")
        print(f"   -> Validare (Cu Promoții >= Fără Promoții): {' VALIDAT' if validare_acord else ' NEVALIDAT'}")

        print("-" * 90)
        if validare_imp and validare_acord:
            print("CONCLUZIE IPOTEZA 4: VALIDATĂ!")
        else:
            print("CONCLUZIE IPOTEZA 4: VALIDARE PARȚIALĂ / NEVALIDATĂ")
        print("=" * 90)

    except Exception as e:
        print(f"Eroare la analiza Ipotezei 4: {e}")


# =======================================================
# IPOTEZA 5: PEARSON & CHI-SQUARE (FRECVENȚĂ & ÎNCREDERE & INFLUENȚĂ)
# =======================================================
def analizeaza_credibilitate_si_frecventa_pearson(file_path):
    try:
        df = pd.read_excel(file_path)
        col_freq = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Frecvența postărilor]'
        col_cred = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Credibilitatea brandului pe social media]'
        col_prez_incredere = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Prezența pe social media îmi arată că un brand este de încredere.]'

        influence_mapping = {'Influență majoră': 5, 'Influență mare': 4, 'Influență moderată': 3, 'Influență redusă': 2,
                             'Fără influență': 1}
        agreement_mapping = {'Acord total': 5, 'Acord': 4, 'Neutru': 3, 'Dezacord': 2, 'Dezacord total': 1}

        df['scor_freq'] = df[col_freq].map(influence_mapping)
        df['scor_cred'] = df[col_cred].map(influence_mapping)
        df['scor_prez'] = df[col_prez_incredere].map(agreement_mapping)

        df['medie_incredere_per_om'] = df[['scor_cred', 'scor_prez']].mean(axis=1)
        r_pearson = df['scor_freq'].corr(df['medie_incredere_per_om'])
        nr_respondenti = len(df)

        if pd.notna(r_pearson) and abs(r_pearson) != 1.0:
            t_stat_abs = abs((r_pearson * math.sqrt(nr_respondenti - 2)) / math.sqrt(1 - r_pearson ** 2))
            p_value = stats.t.sf(t_stat_abs, df=nr_respondenti - 2) * 2
        else:
            p_value = 0.0

        print("=" * 90)
        print("IPOTEZA 5: ANALIZĂ PEARSON (Frecvență Postări & Credibilitate Brand)")
        print("=" * 90)
        print(f"  -> Coeficientul Pearson (r) = {r_pearson:.4f}")
        print(f"  -> Valoarea p (T.DIST.2T) = {p_value:.2e}")
        if p_value < 0.05:
            print("  ->  VALIDAT STATISTIC (p < 0.05). Legătura este semnificativă.")
        else:
            print("  ->  NEVALIDAT STATISTIC (p >= 0.05).")
        print("=" * 90)

    except Exception as e:
        print(f"Eroare: {e}")


def analiza_chi_square_influenta_negativa(file_path):
    try:
        df = pd.read_excel(file_path)
        col_da_nu = 'Ați fost vreodată influențat să NU cumpărați un produs sau serviciu din cauza imaginii slabe sau lipsei de prezență a unei afaceri pe social media?'

        # Numărăm frecvențele pentru TOATE categoriile unice găsite în coloană
        counts = df[col_da_nu].value_counts().sort_index()
        frecvente_observate = counts.values
        categorii = counts.index.tolist()

        # Test Chi-Square
        chi2_stat, p_value = stats.chisquare(f_obs=frecvente_observate)

        print("=" * 90)
        print("IPOTEZA 5: TEST CHI-SQUARE")
        print("=" * 90)
        print(f"Distribuția răspunsurilor:")
        for cat, count in zip(categorii, frecvente_observate):
            print(f"  • {cat}: {count} persoane")

        print(f"\n  -> Valoarea Chi-Square = {chi2_stat:.4f}")
        print(f"  -> Valoarea p = {p_value:.4e}")

        if p_value < 0.05:
            print("  ->  CONCLUZIE: Diferență semnificativă statistic! (Distribuția nu este uniformă).")
        else:
            print("  ->  CONCLUZIE: Nu există diferență semnificativă (Răspunsurile par a fi distribuite uniform).")
        print("=" * 90)

    except Exception as e:
        print(f"Eroare la calculul Chi-Square: {e}")


# ==========================================
# FUNCȚII GENERARE GRAFICE
# ==========================================
def genereaza_grafice_bar_spearman(file_path):
    try:
        df = pd.read_excel(file_path)
        col_calitate = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Calitatea și relevanța conținutului postat (text, imagini, video)]'
        col_claritate = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Claritatea informațiilor oferite]'
        col_satisfactie = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Conținutul social media îmi îmbunătățește satisfacția față de un brand.]'

        cat_importanta = ['Fără influență', 'Influență redusă', 'Influență moderată', 'Influență mare',
                          'Influență majoră']
        cat_acord = ['Dezacord total', 'Dezacord', 'Neutru', 'Acord', 'Acord total']

        counts_calitate = df[col_calitate].value_counts().reindex(cat_importanta).fillna(0)
        counts_claritate = df[col_claritate].value_counts().reindex(cat_importanta).fillna(0)
        counts_satisfactie = df[col_satisfactie].value_counts().reindex(cat_acord).fillna(0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        bars0 = axes[0].barh(counts_calitate.index, counts_calitate.values, color='#9b59b6')
        axes[0].set_title('Importanța calității conținutului', fontsize=12, fontweight='bold')
        axes[0].bar_label(bars0, fmt='%d', padding=5)

        bars1 = axes[1].barh(counts_claritate.index, counts_claritate.values, color='#3498db')
        axes[1].set_title('Importanța clarității informației', fontsize=12, fontweight='bold')
        axes[1].bar_label(bars1, fmt='%d', padding=5)

        bars2 = axes[2].barh(counts_satisfactie.index, counts_satisfactie.values, color='#e74c3c')
        axes[2].set_title('Social media crește satisfacția', fontsize=12, fontweight='bold')
        axes[2].bar_label(bars2, fmt='%d', padding=5)

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eroare grafice bar Spearman: {e}")


def genereaza_pie_chart_descoperire(file_path):
    try:
        df = pd.read_excel(file_path)
        col_descoperire = 'Cât de des descoperiți produse sau servicii noi prin intermediul rețelelor sociale?'

        counts = df[col_descoperire].value_counts()
        labels = [f'{idx}\n({val} pers.)' for idx, val in counts.items()]
        sizes = counts.values

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
        ax.axis('equal')

        plt.title('Frecvența descoperirii de produse/servicii noi pe social media', fontsize=14, fontweight='bold',
                  y=1.05)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eroare pie chart descoperire: {e}")


def genereaza_grafice_ipoteza_2(file_path):
    try:
        df = pd.read_excel(file_path)
        col_recenzii = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Recenziile și mărturiile altor clienți, vizibile pe pagină]'
        col_acord = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Recenziile și comentariile altor clienți îmi influențează decizia de cumpărare.]'
        col_continut = 'Ce tip de conținut vă influențează cel mai mult?'

        cat_importanta = ['Fără influență', 'Influență redusă', 'Influență moderată', 'Influență mare',
                          'Influență majoră']
        cat_acord = ['Dezacord total', 'Dezacord', 'Neutru', 'Acord', 'Acord total']

        counts_imp = df[col_recenzii].value_counts().reindex(cat_importanta).fillna(0)
        counts_acord = df[col_acord].value_counts().reindex(cat_acord).fillna(0)

        target_phrases = ['recenzii ale altor clienți', 'videoclipuri în format lung', 'videoclipuri scurte',
                          'conținut creativ']

        def contine_macar_unul(text):
            if pd.isna(text): return 0
            text_lower = str(text).lower()
            for phrase in target_phrases:
                if phrase in text_lower: return 1
            return 0

        df['flag_continut'] = df[col_continut].apply(contine_macar_unul)
        au_ales = df['flag_continut'].sum()
        nu_au_ales = len(df) - au_ales

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        bars0 = axes[0].bar(counts_imp.index, counts_imp.values, color='#4c72b0')
        axes[0].set_title('Importanța recenziilor', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=25)
        axes[0].bar_label(bars0, fmt='%d', padding=3)

        bars1 = axes[1].bar(counts_acord.index, counts_acord.values, color='#dd8452')
        axes[1].set_title("Recenziile îmi influențează decizia", fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=25)
        axes[1].bar_label(bars1, fmt='%d', padding=3)

        labels_pie = [f'Au ales Recenzii / Video / Creativ\n({au_ales} pers.)',
                      f'Niciuna din cele menționate\n({nu_au_ales} pers.)']
        axes[2].pie([au_ales, nu_au_ales], explode=(0.05, 0), labels=labels_pie, colors=['#55a868', '#c44e52'],
                    autopct='%1.1f%%', shadow=True, startangle=140)
        axes[2].set_title('Preferință pentru conținut vizat', fontsize=12, fontweight='bold')

        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Eroare la generarea graficelor Ipotezei 2: {e}")


def genereaza_pie_charts_contact(file_path):
    try:
        df = pd.read_excel(file_path)
        col_contact1 = 'Ați contactat vreodată o afacere prin social media pentru suport tehnic sau întrebări legate de produse?'
        col_contact2 = 'Ați contactat vreodată o afacere prin social media pentru informații, suport sau reclamații?'

        counts1 = df[col_contact1].value_counts()
        counts2 = df[col_contact2].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        labels1 = [f'{idx}\n({val} pers.)' for idx, val in counts1.items()]
        ax1.pie(counts1.values, labels=labels1, autopct='%1.1f%%', shadow=True, startangle=140)
        ax1.axis('equal')
        ax1.set_title('Contactare pentru suport tehnic', fontsize=13, fontweight='bold')

        labels2 = [f'{idx}\n({val} pers.)' for idx, val in counts2.items()]
        ax2.pie(counts2.values, labels=labels2, autopct='%1.1f%%', shadow=True, startangle=140)
        ax2.axis('equal')
        ax2.set_title('Contactare pentru informații/reclamații', fontsize=13, fontweight='bold')

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eroare la generarea pie charts contact: {e}")


def genereaza_grafice_bar_regresie(file_path):
    try:
        df = pd.read_excel(file_path)
        col_timp = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Timpul de răspuns la mesaje/comentarii]'
        col_prof = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Profesionalismul comunicării]'
        col_sat = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Răspunsurile rapide pe social media îmi cresc satisfacția față de o companie.]'

        cat_importanta = ['Fără influență', 'Influență redusă', 'Influență moderată', 'Influență mare',
                          'Influență majoră']
        cat_acord = ['Dezacord total', 'Dezacord', 'Neutru', 'Acord', 'Acord total']

        counts_timp = df[col_timp].value_counts().reindex(cat_importanta).fillna(0)
        counts_prof = df[col_prof].value_counts().reindex(cat_importanta).fillna(0)
        counts_sat = df[col_sat].value_counts().reindex(cat_acord).fillna(0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        bars0 = axes[0].barh(counts_timp.index, counts_timp.values, color='#abdda4')
        axes[0].set_title('Importanța timpului de răspuns', fontsize=12, fontweight='bold')
        axes[0].bar_label(bars0, fmt='%d', padding=5)

        bars1 = axes[1].barh(counts_prof.index, counts_prof.values, color='#fdae61')
        axes[1].set_title('Importanța profesionalismului', fontsize=12, fontweight='bold')
        axes[1].bar_label(bars1, fmt='%d', padding=5)

        bars2 = axes[2].barh(counts_sat.index, counts_sat.values, color='#2b83ba')
        axes[2].set_title('Răspunsurile rapide cresc satisfacția', fontsize=12, fontweight='bold')
        axes[2].bar_label(bars2, fmt='%d', padding=5)

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eroare la generarea bar charts regresie: {e}")


def genereaza_grafice_ipoteza_4(file_path):
    try:
        df = pd.read_excel(file_path)
        col_oferte_imp = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Ofertele speciale și promoțiile anunțate exclusiv pe social media]'
        col_oferte_acord = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Promoțiile comunicate prin canale sociale mă determină să cumpăr.]'
        col_continut = 'Ce tip de conținut vă influențează cel mai mult?'

        cat_imp = ['Influență majoră', 'Influență mare', 'Influență moderată', 'Influență redusă', 'Fără influență']
        cat_acord = ['Acord total', 'Acord', 'Neutru', 'Dezacord', 'Dezacord total']

        df['flag_promotii'] = df[col_continut].astype(str).str.contains('Promoții și oferte').astype(int)

        # Cele două grupuri opuse
        grup_da = df[df['flag_promotii'] == 1]
        grup_nu = df[df['flag_promotii'] == 0]

        counts_imp_da = grup_da[col_oferte_imp].value_counts().reindex(cat_imp).fillna(0)
        counts_imp_nu = grup_nu[col_oferte_imp].value_counts().reindex(cat_imp).fillna(0)

        counts_acord_da = grup_da[col_oferte_acord].value_counts().reindex(cat_acord).fillna(0)
        counts_acord_nu = grup_nu[col_oferte_acord].value_counts().reindex(cat_acord).fillna(0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Bare alăturate pentru Importanța Ofertelor
        x_imp = np.arange(len(cat_imp))
        width = 0.35

        axes[0].bar(x_imp - width / 2, counts_imp_nu.values, width, label='Grup "Fără Promoții"', color='#cbd5e8')
        axes[0].bar(x_imp + width / 2, counts_imp_da.values, width, label='Grup "Promoții"', color='#377eb8')
        axes[0].set_title('Importanța ofertelor exclusive', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x_imp)
        axes[0].set_xticklabels(cat_imp, rotation=25, ha='right')
        axes[0].legend()

        # Bare alăturate pentru Acord
        x_acord = np.arange(len(cat_acord))
        axes[1].bar(x_acord - width / 2, counts_acord_nu.values, width, label='Grup "Fără Promoții"', color='#fddbc7')
        axes[1].bar(x_acord + width / 2, counts_acord_da.values, width, label='Grup "Promoții"', color='#e41a1c')
        axes[1].set_title('Promoțiile mă determină să cumpăr', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_acord)
        axes[1].set_xticklabels(cat_acord, rotation=25, ha='right')
        axes[1].legend()

        # Pie chart proporții
        au_ales = len(grup_da)
        nu_au_ales = len(grup_nu)
        labels_pie = [f'Au ales "Promoții"\n({au_ales} pers.)', f'Nu au ales\n({nu_au_ales} pers.)']
        axes[2].pie([au_ales, nu_au_ales], explode=(0.05, 0), labels=labels_pie, colors=['#ff9999', '#66b3ff'],
                    autopct='%1.1f%%', shadow=True, startangle=140)
        axes[2].set_title('Câți preferă conținutul cu promoții?', fontsize=12, fontweight='bold')

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eroare la generarea graficelor Ipoteza 4: {e}")


def genereaza_grafice_bar_frecventa_credibilitate(file_path):
    try:
        df = pd.read_excel(file_path)
        col_freq = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Frecvența postărilor]'
        col_cred = 'Evaluați cât de important este ca o afacere să urmeze aceste aspecte pe social media, în ceea ce privește influența deciziei dumneavoastră de cumpărare: [Credibilitatea brandului pe social media]'
        col_prez = 'În ce măsură sunteți de acord cu următoarele afirmații despre comunicarea afacerilor pe social media? [Prezența pe social media îmi arată că un brand este de încredere.]'

        cat_importanta = ['Fără influență', 'Influență redusă', 'Influență moderată', 'Influență mare',
                          'Influență majoră']
        cat_acord = ['Dezacord total', 'Dezacord', 'Neutru', 'Acord', 'Acord total']

        counts_freq = df[col_freq].value_counts().reindex(cat_importanta).fillna(0)
        counts_cred = df[col_cred].value_counts().reindex(cat_importanta).fillna(0)
        counts_prez = df[col_prez].value_counts().reindex(cat_acord).fillna(0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        bars0 = axes[0].barh(counts_freq.index, counts_freq.values, color='#8da0cb')
        axes[0].set_title('Importanța frecvenței postărilor', fontsize=12, fontweight='bold')
        axes[0].bar_label(bars0, fmt='%d', padding=5)

        bars1 = axes[1].barh(counts_cred.index, counts_cred.values, color='#fc8d62')
        axes[1].set_title('Importanța credibilității', fontsize=12, fontweight='bold')
        axes[1].bar_label(bars1, fmt='%d', padding=5)

        bars2 = axes[2].barh(counts_prez.index, counts_prez.values, color='#66c2a5')
        axes[2].set_title('Prezența crește încrederea', fontsize=12, fontweight='bold')
        axes[2].bar_label(bars2, fmt='%d', padding=5)

        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Eroare grafice bar orizontale: {e}")


def genereaza_pie_chart_influenta_negativa(file_path):
    try:
        df = pd.read_excel(file_path)
        col_da_nu = 'Ați fost vreodată influențat să NU cumpărați un produs sau serviciu din cauza imaginii slabe sau lipsei de prezență a unei afaceri pe social media?'

        # Numărăm automat toate variantele de răspuns existente
        counts = df[col_da_nu].value_counts()

        # Creăm etichete care includ și numărul de persoane
        labels = [f'{idx}\n({val} pers.)' for idx, val in counts.items()]
        sizes = counts.values

        # Generăm graficul
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
        ax.axis('equal')

        plt.title('Influența negativă a imaginii slabe', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Eroare pie chart: {e}")


# ==========================================
# INTERFAȚA GRAFICĂ (MODERNĂ pe IPOTEZE)
# ==========================================
class AplicatieAnalizaData:
    def __init__(self, root):
        self.root = root
        self.root.title("Instrument profesional de analiză a datelor - social media")
        self.root.geometry("1100x700")

        self.file_path = None

        # --- Frame Superior (Selecție Fișier) ---
        top_frame = tb.Frame(self.root, padding=15)
        top_frame.pack(fill=tk.X)

        self.lbl_titlu = tb.Label(top_frame, text="Analiză Date Chestionar", font=("Helvetica", 18, "bold"),
                                  bootstyle="primary")
        self.lbl_titlu.pack(pady=(0, 10))

        self.btn_select_file = tb.Button(top_frame, text="Selectează Fișierul Excel", command=self.incarca_fisier,
                                         bootstyle="primary", width=30)
        self.btn_select_file.pack(pady=5)

        self.lbl_file_path = tb.Label(top_frame, text="Niciun fișier selectat", font=("Helvetica", 10, "italic"),
                                      bootstyle="secondary")
        self.lbl_file_path.pack()

        # --- Notebook (Sistemul de Tab-uri Noi) ---
        self.notebook = tb.Notebook(self.root, bootstyle="info")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Creăm Tab-urile pentru Ipoteze
        tab_h1 = tb.Frame(self.notebook, padding=25)
        tab_h2 = tb.Frame(self.notebook, padding=25)
        tab_h3 = tb.Frame(self.notebook, padding=25)
        tab_h4 = tb.Frame(self.notebook, padding=25)
        tab_h5 = tb.Frame(self.notebook, padding=25)

        self.notebook.add(tab_h1, text=" 📌 Ipoteza 1 ")
        self.notebook.add(tab_h2, text=" 📌 Ipoteza 2 ")
        self.notebook.add(tab_h3, text=" 📌 Ipoteza 3 ")
        self.notebook.add(tab_h4, text=" 📌 Ipoteza 4 ")
        self.notebook.add(tab_h5, text=" 📌 Ipoteza 5 ")

        def create_tab_btn(parent, text, command, style="outline-primary"):
            return tb.Button(parent, text=text, command=command, bootstyle=style, width=55)

        # ==========================================
        # IPOTEZA 1
        # ==========================================
        lbl_h1 = tb.Label(tab_h1,
                          text="Ipoteza 1: Calitatea și claritatea conținutului social media sunt asociate pozitiv cu satisfacția utilizatorului",
                          font=("Helvetica", 12, "bold"))
        lbl_h1.pack(pady=(0, 15))

        btn_h1_1 = create_tab_btn(tab_h1, "Analiză Spearman",
                                  self.run_corelatii_spearman, "warning")
        btn_h1_1.pack(pady=8)

        btn_h1_2 = create_tab_btn(tab_h1, "2D Bar Charts pentru întrebările 8 și 10",
                                  self.run_grafice_bar_spearman, "info")
        btn_h1_2.pack(pady=8)

        btn_h1_3 = create_tab_btn(tab_h1, "Pie Chart pentru întrebarea 5",
                                  self.run_pie_chart_descoperire, "info")
        btn_h1_3.pack(pady=8)

        # ==========================================
        # IPOTEZA 2
        # ==========================================
        lbl_h2 = tb.Label(tab_h2,
                          text="Ipoteza 2: Recenziile și conținutul generat de utilizatori influențează decizia de cumpărare mai puternic decât conținutul publicat direct de brand",
                          font=("Helvetica", 12, "bold"), wraplength=1000)
        lbl_h2.pack(pady=(0, 15))

        btn_h2_1 = create_tab_btn(tab_h2, "Analiza rezultatelor",
                                  self.run_analiza_ipoteza_2, "warning")
        btn_h2_1.pack(pady=8)

        btn_h2_2 = create_tab_btn(tab_h2, "Grafice pentru întrebările 7, 8 și 10",
                                  self.run_grafice_ipoteza_2, "info")
        btn_h2_2.pack(pady=8)

        # ==========================================
        # IPOTEZA 3
        # ==========================================
        lbl_h3 = tb.Label(tab_h3,
                          text="Ipoteza 3: Timpul de răspuns rapid și profesionalismul comunicării cresc satisfacția clienților",
                          font=("Helvetica", 12, "bold"), wraplength=1000)
        lbl_h3.pack(pady=(0, 15))

        btn_h3_1 = create_tab_btn(tab_h3, "Regresie Liniară Multiplă",
                                  self.run_regresie_liniara, "warning")
        btn_h3_1.pack(pady=8)

        btn_h3_2 = create_tab_btn(tab_h3, "Pie Charts întrebările 6 și 15", self.run_pie_charts_contact, "info")
        btn_h3_2.pack(pady=8)

        btn_h3_3 = create_tab_btn(tab_h3, "2D Bar Charts pentru întrebările 8 și 10",
                                  self.run_grafice_bar_regresie, "info")
        btn_h3_3.pack(pady=8)

        # ==========================================
        # IPOTEZA 4
        # ==========================================
        lbl_h4 = tb.Label(tab_h4,
                          text="Ipoteza 4: Promoțiile și ofertele comunicate prin social media influențează pozitiv decizia de cumpărare",
                          font=("Helvetica", 12, "bold"), wraplength=1000)
        lbl_h4.pack(pady=(0, 15))

        btn_h4_1 = create_tab_btn(tab_h4, "Analiză Mediane (Grupuri Separate)", self.run_analiza_ipoteza_4,
                                  "warning")
        btn_h4_1.pack(pady=8)

        btn_h4_2 = create_tab_btn(tab_h4, "Grafice întrebările 7, 8 și 10", self.run_grafice_ipoteza_4,
                                  "info")
        btn_h4_2.pack(pady=8)

        # ==========================================
        # IPOTEZA 5
        # ==========================================
        lbl_h5 = tb.Label(tab_h5,
                          text="Ipoteza 5: Prezența constantă și comunicarea profesionistă pe social media cresc încredea în brand și reduc riscul de neîncredere",
                          font=("Helvetica", 12, "bold"), wraplength=1000)
        lbl_h5.pack(pady=(0, 15))

        btn_h5_1 = create_tab_btn(tab_h5, "Analiză Pearson",
                                  self.run_analiza_credibilitate_pearson, "warning")
        btn_h5_1.pack(pady=8)

        btn_h5_1_2 = create_tab_btn(tab_h5, "Analiză de frecvență",
                                    self.run_chi_square_influenta, "warning")
        btn_h5_1_2.pack(pady=8)

        btn_h5_2 = create_tab_btn(tab_h5, "2D Bar Charts pentru întrebările 8 și 10",
                                  self.run_grafice_bar_frecventa_credibilitate, "info")
        btn_h5_2.pack(pady=8)

        btn_h5_3 = create_tab_btn(tab_h5, "Pie Chart pentru întrebarea 12",
                                  self.run_pie_chart_influenta_negativa, "info")
        btn_h5_3.pack(pady=8)

        # --- Frame Inferior (Consolă) ---
        bot_frame = tb.Frame(self.root, padding=20)
        bot_frame.pack(fill=tk.BOTH, expand=True)

        header_console = tb.Frame(bot_frame)
        header_console.pack(fill=tk.X, pady=(0, 5))

        lbl_consola = tb.Label(header_console, text="Consolă Rezultate:", font=("Helvetica", 11, "bold"))
        lbl_consola.pack(side=tk.LEFT)

        self.btn_clear = tb.Button(header_console, text="Curăță Consola", command=self.clear_console,
                                   bootstyle="outline-secondary", width=15)
        self.btn_clear.pack(side=tk.RIGHT)

        self.console_output = ScrolledText(bot_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.console_output.pack(fill=tk.BOTH, expand=True)

        sys.stdout = PrintRedirector(self.console_output)
        print("Aplicația a pornit cu succes. Te rugăm să selectezi fișierul Excel.\n")

    # ==========================================
    # LOGICA BUTOANELOR
    # ==========================================
    def incarca_fisier(self):
        cale = filedialog.askopenfilename(title="Selectează fișierul",
                                          filetypes=(("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")))
        if cale:
            self.file_path = cale
            self.lbl_file_path.config(text=self.file_path, bootstyle="success")
            print(f"[*] Fișier încărcat cu succes:\n{self.file_path}\n")

    def check_file_loaded(self):
        if not self.file_path:
            messagebox.showwarning("Atenție", "Te rog să selectezi întâi un fișier Excel!")
            return False
        return True

    def clear_console(self):
        self.console_output.delete('1.0', tk.END)

    # Ipoteza 1 (Spearman)
    def run_corelatii_spearman(self):
        if self.check_file_loaded(): analizeaza_corelatii_spearman(self.file_path)

    def run_grafice_bar_spearman(self):
        if self.check_file_loaded(): genereaza_grafice_bar_spearman(self.file_path)

    def run_pie_chart_descoperire(self):
        if self.check_file_loaded(): genereaza_pie_chart_descoperire(self.file_path)

    # Ipoteza 2 (Recenzii și Conținut)
    def run_analiza_ipoteza_2(self):
        if self.check_file_loaded(): analiza_ipoteza_2_recenzii(self.file_path)

    def run_grafice_ipoteza_2(self):
        if self.check_file_loaded(): genereaza_grafice_ipoteza_2(self.file_path)

    def run_ttest_influenta(self):
        if self.check_file_loaded(): analiza_t_test_influenta_negativa(self.file_path)

    # Ipoteza 3 (Regresie Multiplă)
    def run_regresie_liniara(self):
        if self.check_file_loaded(): analizeaza_regresie_ordinala(self.file_path)

    def run_pie_charts_contact(self):
        if self.check_file_loaded(): genereaza_pie_charts_contact(self.file_path)

    def run_grafice_bar_regresie(self):
        if self.check_file_loaded(): genereaza_grafice_bar_regresie(self.file_path)

    # Ipoteza 4 (Promoții)
    def run_analiza_ipoteza_4(self):
        if self.check_file_loaded(): analiza_ipoteza_4_promotii(self.file_path)

    def run_grafice_ipoteza_4(self):
        if self.check_file_loaded(): genereaza_grafice_ipoteza_4(self.file_path)

    # Ipoteza 5 (Pearson & Chi-Square)
    def run_analiza_credibilitate_pearson(self):
        if self.check_file_loaded(): analizeaza_credibilitate_si_frecventa_pearson(self.file_path)

    def run_chi_square_influenta(self):
        if self.check_file_loaded(): analiza_chi_square_influenta_negativa(self.file_path)

    def run_grafice_bar_frecventa_credibilitate(self):
        if self.check_file_loaded(): genereaza_grafice_bar_frecventa_credibilitate(self.file_path)

    def run_pie_chart_influenta_negativa(self):
        if self.check_file_loaded(): genereaza_pie_chart_influenta_negativa(self.file_path)


if __name__ == "__main__":
    root = tb.Window(themename="yeti")
    app = AplicatieAnalizaData(root)
    root.mainloop()
