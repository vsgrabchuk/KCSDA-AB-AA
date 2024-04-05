import pandas as pd
import numpy as np
from scipy import stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def synth_test(
    s1,
    s2,
    ss_percent=10,
    n_simulations=10000,
    alpha=0.05,
    test='t',
    print_info=True,
    bins=20
):
    '''
    Функция, производящая синтетический тест на 2х выборках
    Синтетический тест - сравнение множества подвыборок без повторений
    
    Parameters
    ----------
    s1: pandas.Series
        Выборка 1 (sample)
    s2: pandas.Series
        Выборка 2
    ss_percent: float, default 10
        Процент от выборки (min размера) для составления подвыборки (subsample)
    n_simulations: int, default 10000
        Количество симуляций
    test: str, default 't'
        Статистический тест
        't' - t-тест
        'u' - тест Манна-Уитни
    alpha: float, default 0.05
        Уровень значимости
    print_info: bool, default True
        Флаг отображения информации (текст+графики)
    bins: int, default 20
        Количество bin'ов для отображения гистограммы
    
    Returns
    -------
    fpr: float
        FPR
    '''
    n_s_min = min(len(s1), len(s2))  # Минимальный размер из выборок
    n_ss = round(n_s_min * ss_percent / 100)  # Количество элементов в подвыборке
    
    p_vals = []  # Список с p-values
    
    # Цикл симуляций
    my_range = range(n_simulations)
    if print_info:
        my_range = tqdm(my_range)
    for i in my_range:
        ss1 = s1.sample(n_ss, replace=False)
        ss2 = s2.sample(n_ss, replace=False)
        # Сравнение подвыборок
        if test == 't':  # t-тест с поправкой Уэлча
            test_res = st.ttest_ind(ss1, ss2, equal_var=False)
        elif test == 'u':  # U-тест
            test_res = st.mannwhitneyu(ss1, ss2)
            
        p_vals.append(test_res[1]) # Сохраняем p-value
        
    # FPR
    fpr = sum(np.array(p_vals) < alpha) / n_simulations
    
    # Визулаилзация распределения p-values
    if print_info:
        print('min sample size:', n_s_min)
        print('synthetic subsample size:', n_ss)
        plt.style.use('ggplot')
        _, _, bars = plt.hist(p_vals, bins=bins)
        for bar in bars:
            bar.set_edgecolor('black')
            if bar.get_x() < alpha:
                # Статзначимая разница
                bar.set_facecolor('#f74a64')
            else:
                bar.set_facecolor('grey')
        plt.xlabel('p-values')
        plt.ylabel('frequency')
        plt.title(f"FPR: {fpr}") 
        
        sm.qqplot(np.array(p_vals), dist=st.uniform, line="45")

        plt.show()

    return fpr
