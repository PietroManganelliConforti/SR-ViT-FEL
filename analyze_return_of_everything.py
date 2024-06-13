import csv
from functools import partial
from itertools import chain
from pprint import pprint
from typing import List, NamedTuple, Tuple
import decimal as dec

import numpy as np


class Sci(NamedTuple):
    """Exact representation for rational numbers in scientific notation.
    
    Attributes:
        significand: The significand of the number, represented as a Decimal to
            avoid floating point errors.
        exponent: The (integer) power of 10 that multiplies the significand,
            i.e. the order of magnitude.
    """

    significand: dec.Decimal
    exponent: int

    @staticmethod
    def from_number(x):
        """Create a Sci object from a number.
        
        The exponent is chosen to be the power of 10 corresponding to the
        first non-zero digit of the decimal representation of the number, and
        the significand is the number divided by 10^exponent.
        """
        if isinstance(x, Sci):
            x = x.decimal()
        x = dec.Decimal(x)
        om = x.adjusted()
        x *= dec.Decimal(10)**(-om)
        return Sci(x, om)

    def __float__(self):
        return float(self.significand) * 10**self.exponent
    
    def decimal(self):
        # TODO this sometimes adds trailing zeros
        return self.significand.scaleb(self.exponent)

    def __str__(self):
        return f'{self.significand}e{self.exponent}'

    def round(self, places=2, rounding=dec.ROUND_HALF_UP):
        """Round the significand to the given number of decimal places.
        
        More specifically, the significand is rounded to the nearest multiple
        of 10^-(places-1).
        """
        significand = self.significand.quantize(dec.Decimal(10)**(-(places-1)), rounding=rounding)
        return Sci(significand, self.exponent)

    def adjust(self, exponent: int) -> 'Sci':
        # Adjust the significand by dividing for the exponent difference.
        to_other = self.exponent - exponent
        return Sci(self.significand * dec.Decimal(10)**to_other, exponent)

    def adjust_like(self, other: 'Sci') -> 'Sci':
        return self.adjust(other.exponent)

    def explicit(self) -> str:
        sign, digits, exponent = self.decimal().as_tuple()
        explicit = '' if sign == 0 else '-'
        if len(digits) + exponent < 0:
            explicit += '0.'
            explicit += '0' * (-(len(digits) + exponent))
        for i, d in enumerate(digits):
            if i == len(digits) + exponent:
                explicit += '.'
            explicit += str(d)
        if exponent > 0:
            explicit += '0' * exponent
        if explicit.startswith('-.'):
            explicit = '-0.' + explicit[2:]
        if explicit.startswith('.'):
            explicit = '0' + explicit
        return explicit

    # if exponent is None, chooses the exponent that appears the most
    def align(*xs, exponent=None):
        xs = tuple(map(Sci.from_number, xs))
        if exponent is None:
            exponent = max(set(x.exponent for x in xs), key=lambda e: sum(x.exponent == e for x in xs))
        return (x.adjust(exponent) for x in xs)


class tex:

    downarrow = '\\downarrow'
    pm = '\\pm'

    @staticmethod
    def bold(s: str) -> str:
        return f'\\textbf{{{s}}}'

    @staticmethod
    def words(*words) -> str:
        return ' '.join(words)

    @staticmethod
    def math(s: str, inline=True) -> str:
        assert inline
        return f'${s}$'

    # other values for infix: ' \\cdot 10^', ' \\times 10^'
    @staticmethod
    def real(x: Sci, infix=None) -> str:
        explicit = x.explicit()
        if infix is None:
            exponential = str(x)
        else:
            exponential = f'{x.significand}{infix}{x.exponent}'
        if len(explicit) < len(exponential):
            return explicit
        return exponential

    @staticmethod
    def best_exponent(*xs) -> int:
        xs = tuple(Sci.from_number(x) for x in xs)
        exponents = [x.exponent for x in xs]
        a = min(exponents)
        b = max(exponents)
        lens = [ max(len(tex.real(x.adjust(e))) for x in xs) for e in range(a, b+1)]
        return min(range(a, b+1), key=lambda e: lens[e-a])

    @staticmethod
    def tabular(columns: str, *rows: str) -> str:
        header = r"\begin{tabular}" + "{" + columns + "}\n"
        footer = r"\end{tabular}" + "\n"
        return header + '\n'.join(rows) + footer

    @staticmethod
    # rule as in ruler, not as in regulation
    def rule(name) -> str:
        if name not in {'mid', 'bottom', 'top'}:
            raise ValueError(f'Invalid rule name: {name}')
        return f'\\{name}rule'

    @staticmethod
    def cells(*cells_) -> str:
        return ' & '.join(cells_) + ' \\\\'

    @staticmethod
    def map_column(column: int, f, rows: List[List[str]]) -> List[List[str]]:
        col = [row[column] for row in rows]
        col = f(col)
        for i, row in enumerate(rows):
            row[column] = col[i]
        return rows



# Rows format examples:
# [0.52369882],[1.00851274],[0.00600504],2D,False,False,False,"[([0.6615703701972961], [0.32831859588623047], [0.004906835965812206]), ([0.5819323658943176], [0.5005238056182861], [0.004854969214648008]), ([1.726336121559143], [0.5681396722793579], [0.007701393682509661]), ([1.305978775024414], [0.6371706128120422], [0.006951622664928436]), ([0.7667460441589355], [0.5843414068222046], [0.00561036029830575])]","results_24_04/_cross_val_5di5_2D_2D_forecasting_lstm_CO(GT)_ricker_8_['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']/"
# [0.43787231],[0.85119252],[0.00555139],2D_ViT_SR_feat_in,True,False,False,"[([0.5758355259895325], [0.2968023419380188], [0.004672243259847164]), ([0.5654610395431519], [0.4527936577796936], [0.004671864677220583]), ([1.488579511642456], [0.40863776206970215], [0.00715209636837244]), ([0.9715458750724792], [0.5176060199737549], [0.0060907406732439995]), ([0.6545406579971313], [0.5135217905044556], [0.005170024931430817])]","results_24_04/_cross_val_5di5_2D_ViT_SR_feat_in_2D_forecasting_lstm_CO(GT)_ricker_8_['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']_augmented_aug_type_noise1/"
# [0.46253561],[0.90203047],[0.0055803],2D_ViT_parallel_SR,False,False,True,"[([0.6599553227424622], [0.3139924705028534], [0.004858754575252533]), ([0.559723436832428], [0.44197648763656616], [0.004582221154123545]), ([1.335585355758667], [0.4618488848209381], [0.006707665044814348]), ([1.2938846349716187], [0.5840272903442383], [0.006709683686494827]), ([0.661003589630127], [0.5108329057693481], [0.005043159704655409])]","results_24_04/_cross_val_5di5_2D_ViT_parallel_SR_2D_forecasting_lstm_CO(GT)_ricker_8_['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']_freezed/"

class Run(NamedTuple):
    mape_mean: Sci
    mse_mean: Sci
    mase_mean: Sci
    arch: str
    wavelet: str
    augmented: bool
    freezed: bool
    pretrained: bool
    results: Tuple[np.ndarray, np.ndarray, np.ndarray]

def parse_row(row):
    results = tuple(zip(*eval(row[7])))
    results = tuple([np.array([x[0] for x in xs]) for xs in results])
    sci = lambda x: Sci.from_number(x).round(places=3)
    run = Run(
        mape_mean = sci(eval(row[0])[0]),
        mse_mean = sci(eval(row[1])[0]),
        mase_mean = sci(eval(row[2])[0]),
        arch = row[3],
        wavelet = 'morlet2' if 'morlet2' in row[8] else 'morlet' if 'morlet' in row[8] else 'ricker',
        augmented = eval(row[4]),
        pretrained = eval(row[5]),
        freezed = eval(row[6]),
        results = results,
        # path = row[8],
    )
    assert run.wavelet != 'ricker' or 'ricker' in row[8]
    return run


def main():

    with open("return_of_everything.csv", "r") as file:
        csv_reader = csv.reader(file)
        runs = list(csv_reader)

    runs = [parse_row(row) for row in runs]
    # pprint(runs)

    assert all(not run.pretrained for run in runs)

    # runs = [ run for run in runs if not run.freezed and not run.augmented ]
    runs = [ run for run in runs if run.freezed and not run.augmented ]

    header = tex.cells(
        tex.bold('Test Name'),
        tex.bold('Wavelet'),
        tex.words(tex.bold('RE'), tex.math(tex.downarrow)),
        tex.words(tex.bold('Loss'), tex.math(tex.downarrow)),
        tex.words(tex.bold('MASE'), tex.math(tex.downarrow)),
    )

    # mape_exponent = tex.best_exponent(*[run.mape_mean for run in runs])
    # mse_exponent = tex.best_exponent(*[run.mse_mean for run in runs])
    # mase_exponent = tex.best_exponent(*[run.mase_mean for run in runs])

    def map_metric(xs):
        means, stds = list(zip(*xs))
        means_stds = list(Sci.align(*chain(means, stds)))
        means_stds = [x.round(places=3) for x in means_stds]
        means = means_stds[:len(means)]
        stds = means_stds[len(means):]
        return list(tex.words(tex.real(mean), tex.math(tex.pm), tex.real(std)) for mean, std in zip(means, stds))

    def get_metric(run: Run, metric):
        mean = getattr(run, f'{metric}_mean')
        results = run.results[{ 'mape': 0, 'mse': 1, 'mase': 2 }[metric]]
        std = np.std(results)
        return mean, std

    ricker = [
        [
            run.arch.replace('_', '\\_'),
            'Ricker',
            get_metric(run, 'mape'),
            get_metric(run, 'mse'),
            get_metric(run, 'mase'),
        ]
        for run in runs if run.wavelet == 'ricker'
    ]

    tex.map_column(2, map_metric, ricker)
    tex.map_column(3, map_metric, ricker)
    tex.map_column(4, map_metric, ricker)

    morlet2 = [
        [
            run.arch.replace('_', '\\_'),
            'Morlet2',
            get_metric(run, 'mape'),
            get_metric(run, 'mse'),
            get_metric(run, 'mase'),
        ]
        for run in runs if run.wavelet == 'morlet2'
    ]

    tex.map_column(2, map_metric, morlet2)
    tex.map_column(3, map_metric, morlet2)
    tex.map_column(4, map_metric, morlet2)


    tabular = tex.tabular('lcccc',
        header,
        tex.rule('top'),
        *(tex.cells(*r) for r in ricker),
        tex.rule('mid'),
        *(tex.cells(*r) for r in morlet2),
        tex.rule('bottom'),
    )

    print(tabular)

    # print("Augmented:")
    # for run in aug:
    #     print('  ', run.arch)

    # print("Freezed:")
    # for run in freezed:
    #     print('  ', run.arch)

    # print("Other:")
    # for run in other:
    #     print('  ', run.arch)





if __name__ == "__main__":
    main()
