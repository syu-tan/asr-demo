# -*- coding: utf-8 -*-

#
# メルフィルタバンク特徴の計算を行います．
#

# wavデータを読み込むためのモジュール(wave)をインポート
import wave

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# os, sysモジュールをインポート
import os
import sys


class FeatureExtractor:
    """ 特徴量(FBANK, MFCC)を抽出するクラス
    sample_frequency: 入力波形のサンプリング周波数 [Hz]
    frame_length: フレームサイズ [ミリ秒]
    frame_shift: 分析間隔(フレームシフト) [ミリ秒]
    num_mel_bins: メルフィルタバンクの数(=FBANK特徴の次元数)
    num_ceps: MFCC特徴の次元数(0次元目を含む)
    lifter_coef: リフタリング処理のパラメータ
    low_frequency: 低周波数帯域除去のカットオフ周波数 [Hz]
    high_frequency: 高周波数帯域除去のカットオフ周波数 [Hz]
    dither: ディザリング処理のパラメータ(雑音の強さ)
    """

    # クラスを呼び出した時点で最初に1回実行される関数
    def __init__(
        self,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
        num_mel_bins=23,
        num_ceps=13,
        lifter_coef=22,
        low_frequency=20,
        high_frequency=8000,
        dither=1.0,
    ):
        # サンプリング周波数[Hz]
        self.sample_freq = sample_frequency
        # 窓幅をミリ秒からサンプル数へ変換
        self.frame_size = int(sample_frequency * frame_length * 0.001)
        # フレームシフトをミリ秒からサンプル数へ変換
        self.frame_shift = int(sample_frequency * frame_shift * 0.001)
        # メルフィルタバンクの数
        self.num_mel_bins = num_mel_bins
        # MFCCの次元数(0次含む)
        self.num_ceps = num_ceps
        # リフタリングのパラメータ
        self.lifter_coef = lifter_coef
        # 低周波数帯域除去のカットオフ周波数[Hz]
        self.low_frequency = low_frequency
        # 高周波数帯域除去のカットオフ周波数[Hz]
        self.high_frequency = high_frequency
        # ディザリング係数
        self.dither_coef = dither

        # FFTのポイント数 = 窓幅以上の2のべき乗
        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        # メルフィルタバンクを作成する
        self.mel_filter_bank = self.MakeMelFilterBank()

        # 離散コサイン変換(DCT)の基底行列を作成する
        self.dct_matrix = self.MakeDCTMatrix()

        # リフタ(lifter)を作成する
        self.lifter = self.MakeLifter()

    def Herz2Mel(self, herz):
        """ 周波数をヘルツからメルに変換する
        """
        return 1127.0 * np.log(1.0 + herz / 700)

    def MakeMelFilterBank(self):
        """ メルフィルタバンクを作成する
        """
        # メル軸での最大周波数
        mel_high_freq = self.Herz2Mel(self.high_frequency)
        # メル軸での最小周波数
        mel_low_freq = self.Herz2Mel(self.low_frequency)
        # 最小から最大周波数まで，
        # メル軸上での等間隔な周波数を得る
        mel_points = np.linspace(mel_low_freq, mel_high_freq, self.num_mel_bins + 2)

        # パワースペクトルの次元数 = FFTサイズ/2+1
        # ※Kaldiの実装ではナイキスト周波数成分(最後の+1)は
        # 捨てているが，本実装では捨てずに用いている
        dim_spectrum = int(self.fft_size / 2) + 1

        # メルフィルタバンク(フィルタの数 x スペクトルの次元数)
        mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))
        for m in range(self.num_mel_bins):
            # 三角フィルタの左端，中央，右端のメル周波数
            left_mel = mel_points[m]
            center_mel = mel_points[m + 1]
            right_mel = mel_points[m + 2]
            # パワースペクトルの各ビンに対応する重みを計算する
            for n in range(dim_spectrum):
                # 各ビンに対応するヘルツ軸周波数を計算
                freq = 1.0 * n * self.sample_freq / 2 / dim_spectrum
                # メル周波数に変換
                mel = self.Herz2Mel(freq)
                # そのビンが三角フィルタの範囲に入っていれば，重みを計算
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filter_bank[m][n] = weight

        return mel_filter_bank

    def ExtractWindow(self, waveform, start_index, num_samples):
        """
        1フレーム分の波形データを抽出し，前処理を実施する．
        また，対数パワーの値も計算する
        """
        # waveformから，1フレーム分の波形を抽出する
        window = waveform[start_index : start_index + self.frame_size].copy()

        # ディザリングを行う
        # (-dither_coef～dither_coefの一様乱数を加える)
        if self.dither_coef > 0:
            window = (
                window
                + np.random.rand(self.frame_size) * (2 * self.dither_coef)
                - self.dither_coef
            )

        # 直流成分をカットする
        window = window - np.mean(window)

        # 以降の処理を行う前に，パワーを求める
        power = np.sum(window ** 2)
        # 対数計算時に-infが出力されないよう，フロアリング処理を行う
        if power < 1e-10:
            power = 1e-10
        # 対数をとる
        log_power = np.log(power)

        # プリエンファシス(高域強調)
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window, np.array([1.0, -0.97]), mode="same")
        # numpyの畳み込みでは0番目の要素が処理されない
        # (window[i-1]が存在しないので)ため，
        # window[0-1]をwindow[0]で代用して処理する
        window[0] -= 0.97 * window[0]

        # hamming窓をかける
        # hamming[i] = 0.54 - 0.46 * np.cos(2*np.pi*i / (self.frame_size - 1))
        window *= np.hamming(self.frame_size)

        return window, log_power

    def ComputeFBANK(self, waveform):
        """メルフィルタバンク特徴(FBANK)を計算する
        出力1: fbank_features: メルフィルタバンク特徴
        出力2: log_power: 対数パワー値(MFCC抽出時に使用)
        """
        # 波形データの総サンプル数
        num_samples = np.size(waveform)
        # 特徴量の総フレーム数を計算する
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        # メルフィルタバンク特徴
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        # 対数パワー(MFCC特徴を求める際に使用する)
        log_power = np.zeros(num_frames)

        # 1フレームずつ特徴量を計算する
        for frame in range(num_frames):
            # 分析の開始位置は，フレーム番号(0始まり)*フレームシフト
            start_index = frame * self.frame_shift
            # 1フレーム分の波形を抽出し，前処理を実施する．
            # また対数パワーの値も得る
            window, log_pow = self.ExtractWindow(waveform, start_index, num_samples)

            # 高速フーリエ変換(FFT)を実行
            spectrum = np.fft.fft(window, n=self.fft_size)
            # FFT結果の右半分(負の周波数成分)を取り除く
            # ※Kaldiの実装ではナイキスト周波数成分(最後の+1)は捨てているが，
            # 本実装では捨てずに用いている
            spectrum = spectrum[: int(self.fft_size / 2) + 1]

            # パワースペクトルを計算する
            spectrum = np.abs(spectrum) ** 2

            # メルフィルタバンクを畳み込む
            fbank = np.dot(spectrum, self.mel_filter_bank.T)

            # 対数計算時に-infが出力されないよう，フロアリング処理を行う
            fbank[fbank < 0.1] = 0.1

            # 対数をとってfbank_featuresに加える
            fbank_features[frame] = np.log(fbank)

            # 対数パワーの値をlog_powerに加える
            log_power[frame] = log_pow

        return fbank_features, log_power

    def MakeDCTMatrix(self):
        """ 離散コサイン変換(DCT)の基底行列を作成する
        """
        N = self.num_mel_bins
        # DCT基底行列 (基底数(=MFCCの次元数) x FBANKの次元数)
        dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
        for k in range(self.num_ceps):
            if k == 0:
                dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
            else:
                dct_matrix[k] = np.sqrt(2 / N) * np.cos(
                    ((2.0 * np.arange(N) + 1) * k * np.pi) / (2 * N)
                )

        return dct_matrix

    def MakeLifter(self):
        """ リフタを計算する
        """
        Q = self.lifter_coef
        I = np.arange(self.num_ceps)
        lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
        return lifter

    def ComputeMFCC(self, waveform):
        """ MFCCを計算する
        """
        # FBANKおよび対数パワーを計算する
        fbank, log_power = self.ComputeFBANK(waveform)

        # DCTの基底行列との内積により，DCTを実施する
        mfcc = np.dot(fbank, self.dct_matrix.T)

        # リフタリングを行う
        mfcc *= self.lifter

        # MFCCの0次元目を，前処理をする前の波形の対数パワーに置き換える
        mfcc[:, 0] = log_power

        return mfcc


# サンプリング周波数 [Hz]
sample_frequency = 16000
# フレーム長 [ミリ秒]
frame_length = 25
# フレームシフト [ミリ秒]
frame_shift = 10
# 低周波数帯域除去のカットオフ周波数 [Hz]
low_frequency = 20
# 高周波数帯域除去のカットオフ周波数 [Hz]
high_frequency = sample_frequency / 2
# メルフィルタバンク特徴の次元数
num_mel_bins = 40
# ディザリングの係数
dither = 1.0

# 乱数シードの設定(ディザリング処理結果の再現性を担保)
np.random.seed(seed=0)

# 特徴量抽出クラスを呼び出す
feat_extractor = FeatureExtractor(
    sample_frequency=sample_frequency,
    frame_length=frame_length,
    frame_shift=frame_shift,
    num_mel_bins=num_mel_bins,
    low_frequency=low_frequency,
    high_frequency=high_frequency,
    dither=dither,
)

from pydub import AudioSegment as am

def preprocess(wav_path):

    processed_path = "./tmp/processed.wav"

    
    sound = am.from_file(wav_path, format='wav', frame_rate=44100)
    sound = sound.set_frame_rate(16000)
    sound.export(processed_path, format='wav')

    with wave.open(processed_path) as wav:
        # サンプリング周波数のチェック
        print(f"{wav.getframerate()=} {wav.getnchannels()=}")
        if wav.getframerate() != sample_frequency:
            sys.stderr.write(
                "The expected \
                sampling rate is 16000.\n"
            )
            exit(1)
        # wavファイルが1チャネル(モノラル)
        # データであることをチェック
        if wav.getnchannels() != 1:
            sys.stderr.write(
                "This program \
                supports monaural wav file only.\n"
            )
            exit(1)

        # wavデータのサンプル数
        num_samples = wav.getnframes()

        # wavデータを読み込む
        waveform = wav.readframes(num_samples)

        # 読み込んだデータはバイナリ値
        # (16bit integer)なので，数値(整数)に変換する
        waveform = np.frombuffer(waveform, dtype=np.int16)

        # FBANKを計算する(log_power:対数パワー情報も
        # 出力されるが，ここでは使用しない)
        fbank, log_power = feat_extractor.ComputeFBANK(waveform)

        # 特徴量のフレーム数と次元数を取得
        (num_frames, num_dims) = np.shape(fbank)

    # データをfloat32形式に変換
    fbank = fbank.astype(np.float32)
    return fbank, num_frames, num_dims


import re

"""かな⇔ローマ字を変換する

https://mohayonao.hatenadiary.org/entry/20091129/1259505966
"""

def _make_kana_convertor():
    """ひらがな⇔カタカナ変換器を作る"""
    kata = {
        'ア':'あ', 'イ':'い', 'ウ':'う', 'エ':'え', 'オ':'お',
        'カ':'か', 'キ':'き', 'ク':'く', 'ケ':'け', 'コ':'こ',
        'サ':'さ', 'シ':'し', 'ス':'す', 'セ':'せ', 'ソ':'そ',
        'タ':'た', 'チ':'ち', 'ツ':'つ', 'テ':'て', 'ト':'と',
        'ナ':'な', 'ニ':'に', 'ヌ':'ぬ', 'ネ':'ね', 'ノ':'の',
        'ハ':'は', 'ヒ':'ひ', 'フ':'ふ', 'ヘ':'へ', 'ホ':'ほ',
        'マ':'ま', 'ミ':'み', 'ム':'む', 'メ':'め', 'モ':'も',
        'ヤ':'や', 'ユ':'ゆ', 'ヨ':'よ', 'ラ':'ら', 'リ':'り',
        'ル':'る', 'レ':'れ', 'ロ':'ろ', 'ワ':'わ', 'ヲ':'を',
        'ン':'ん',
        
        'ガ':'が', 'ギ':'ぎ', 'グ':'ぐ', 'ゲ':'げ', 'ゴ':'ご',
        'ザ':'ざ', 'ジ':'じ', 'ズ':'ず', 'ゼ':'ぜ', 'ゾ':'ぞ',
        'ダ':'だ', 'ヂ':'ぢ', 'ヅ':'づ', 'デ':'で', 'ド':'ど',
        'バ':'ば', 'ビ':'び', 'ブ':'ぶ', 'ベ':'べ', 'ボ':'ぼ',
        'パ':'ぱ', 'ピ':'ぴ', 'プ':'ぷ', 'ペ':'ぺ', 'ポ':'ぽ',
        
        'ァ':'ぁ', 'ィ':'ぃ', 'ゥ':'ぅ', 'ェ':'ぇ', 'ォ':'ぉ',
        'ャ':'ゃ', 'ュ':'ゅ', 'ョ':'ょ',
        'ヴ':'&#12436;', 'ッ':'っ', 'ヰ':'ゐ', 'ヱ':'ゑ',
        }
    
    # ひらがな → カタカナ のディクショナリをつくる
    hira = dict([(v, k) for k, v in kata.items() ])
    
    re_hira2kata = re.compile("|".join(map(re.escape, hira)))
    re_kata2hira = re.compile("|".join(map(re.escape, kata)))
    
    def _hiragana2katakana(text):
        return re_hira2kata.sub(lambda x: hira[x.group(0)], text)
    
    def _katakana2hiragana(text):
        return re_kata2hira.sub(lambda x: kata[x.group(0)], text)
    
    return (_hiragana2katakana, _katakana2hiragana)


hiragana2katakana, katakana2hiragana = _make_kana_convertor()

################################################################################

def _make_romaji_convertor():
    """ローマ字⇔かな変換器を作る"""
    master = {
        'a'  :'ア', 'i'  :'イ', 'u'  :'ウ', 'e'  :'エ', 'o'  :'オ',
        'ka' :'カ', 'ki' :'キ', 'ku' :'ク', 'ke' :'ケ', 'ko' :'コ',
        'sa' :'サ', 'shi':'シ', 'su' :'ス', 'se' :'セ', 'so' :'ソ',
        'ta' :'タ', 'chi':'チ', 'tu' :'ツ', 'te' :'テ', 'to' :'ト',
        'na' :'ナ', 'ni' :'ニ', 'nu' :'ヌ', 'ne' :'ネ', 'no' :'ノ',
        'ha' :'ハ', 'hi' :'ヒ', 'fu' :'フ', 'he' :'ヘ', 'ho' :'ホ',
        'ma' :'マ', 'mi' :'ミ', 'mu' :'ム', 'me' :'メ', 'mo' :'モ',
        'ya' :'ヤ', 'yu' :'ユ', 'yo' :'ヨ',
        'ra' :'ラ', 'ri' :'リ', 'ru' :'ル', 're' :'レ', 'ro' :'ロ',
        'wa' :'ワ', 'wo' :'ヲ', 'n'  :'ン', 'vu' :'ヴ',
        'ga' :'ガ', 'gi' :'ギ', 'gu' :'グ', 'ge' :'ゲ', 'go' :'ゴ',
        'za' :'ザ', 'ji' :'ジ', 'zu' :'ズ', 'ze' :'ゼ', 'zo' :'ゾ',
        'da' :'ダ', 'di' :'ヂ', 'du' :'ヅ', 'de' :'デ', 'do' :'ド',
        'ba' :'バ', 'bi' :'ビ', 'bu' :'ブ', 'be' :'ベ', 'bo' :'ボ',
        'pa' :'パ', 'pi' :'ピ', 'pu' :'プ', 'pe' :'ペ', 'po' :'ポ',
        
        'kya':'キャ', 'kyi':'キィ', 'kyu':'キュ', 'kye':'キェ', 'kyo':'キョ',
        'gya':'ギャ', 'gyi':'ギィ', 'gyu':'ギュ', 'gye':'ギェ', 'gyo':'ギョ',
        'sha':'シャ',               'shu':'シュ', 'she':'シェ', 'sho':'ショ',
        'ja' :'ジャ',               'ju' :'ジュ', 'je' :'ジェ', 'jo' :'ジョ',
        'cha':'チャ',               'chu':'チュ', 'che':'チェ', 'cho':'チョ',
        'dya':'ヂャ', 'dyi':'ヂィ', 'dyu':'ヂュ', 'dhe':'デェ', 'dyo':'ヂョ',
        'nya':'ニャ', 'nyi':'ニィ', 'nyu':'ニュ', 'nye':'ニェ', 'nyo':'ニョ',
        'hya':'ヒャ', 'hyi':'ヒィ', 'hyu':'ヒュ', 'hye':'ヒェ', 'hyo':'ヒョ',
        'bya':'ビャ', 'byi':'ビィ', 'byu':'ビュ', 'bye':'ビェ', 'byo':'ビョ',
        'pya':'ピャ', 'pyi':'ピィ', 'pyu':'ピュ', 'pye':'ピェ', 'pyo':'ピョ',
        'mya':'ミャ', 'myi':'ミィ', 'myu':'ミュ', 'mye':'ミェ', 'myo':'ミョ',
        'rya':'リャ', 'ryi':'リィ', 'ryu':'リュ', 'rye':'リェ', 'ryo':'リョ',
        'fa' :'ファ', 'fi' :'フィ',               'fe' :'フェ', 'fo' :'フォ',
        'wi' :'ウィ', 'we' :'ウェ', 
        'va' :'ヴァ', 'vi' :'ヴィ', 've' :'ヴェ', 'vo' :'ヴォ',
        
        'kwa':'クァ', 'kwi':'クィ', 'kwu':'クゥ', 'kwe':'クェ', 'kwo':'クォ',
        'kha':'クァ', 'khi':'クィ', 'khu':'クゥ', 'khe':'クェ', 'kho':'クォ',
        'gwa':'グァ', 'gwi':'グィ', 'gwu':'グゥ', 'gwe':'グェ', 'gwo':'グォ',
        'gha':'グァ', 'ghi':'グィ', 'ghu':'グゥ', 'ghe':'グェ', 'gho':'グォ',
        'swa':'スァ', 'swi':'スィ', 'swu':'スゥ', 'swe':'スェ', 'swo':'スォ',
        'swa':'スァ', 'swi':'スィ', 'swu':'スゥ', 'swe':'スェ', 'swo':'スォ',
        'zwa':'ズヮ', 'zwi':'ズィ', 'zwu':'ズゥ', 'zwe':'ズェ', 'zwo':'ズォ',
        'twa':'トァ', 'twi':'トィ', 'twu':'トゥ', 'twe':'トェ', 'two':'トォ',
        'dwa':'ドァ', 'dwi':'ドィ', 'dwu':'ドゥ', 'dwe':'ドェ', 'dwo':'ドォ',
        'mwa':'ムヮ', 'mwi':'ムィ', 'mwu':'ムゥ', 'mwe':'ムェ', 'mwo':'ムォ',
        'bwa':'ビヮ', 'bwi':'ビィ', 'bwu':'ビゥ', 'bwe':'ビェ', 'bwo':'ビォ',
        'pwa':'プヮ', 'pwi':'プィ', 'pwu':'プゥ', 'pwe':'プェ', 'pwo':'プォ',
        'phi':'プィ', 'phu':'プゥ', 'phe':'プェ', 'pho':'フォ',
        }
    
    
    romaji_asist = {
        'si' :'シ'  , 'ti' :'チ'  , 'hu' :'フ' , 'zi':'ジ',
        'sya':'シャ', 'syu':'シュ', 'syo':'ショ',
        'tya':'チャ', 'tyu':'チュ', 'tyo':'チョ',
        'cya':'チャ', 'cyu':'チュ', 'cyo':'チョ',
        'jya':'ジャ', 'jyu':'ジュ', 'jyo':'ジョ', 'pha':'ファ', 
        'qa' :'クァ', 'qi' :'クィ', 'qu' :'クゥ', 'qe' :'クェ', 'qo':'クォ',
        
        'ca' :'カ', 'ci':'シ', 'cu':'ク', 'ce':'セ', 'co':'コ',
        'la' :'ラ', 'li':'リ', 'lu':'ル', 'le':'レ', 'lo':'ロ',

        'mb' :'ム', 'py':'パイ', 'tho': 'ソ', 'thy':'ティ', 'oh':'オウ',
        'by':'ビィ', 'cy':'シィ', 'dy':'ディ', 'fy':'フィ', 'gy':'ジィ',
        'hy':'シー', 'ly':'リィ', 'ny':'ニィ', 'my':'ミィ', 'ry':'リィ',
        'ty':'ティ', 'vy':'ヴィ', 'zy':'ジィ',
        
        'b':'ブ', 'c':'ク', 'd':'ド', 'f':'フ'  , 'g':'グ', 'h':'フ', 'j':'ジ',
        'k':'ク', 'l':'ル', 'm':'ム', 'p':'プ'  , 'q':'ク', 'r':'ル', 's':'ス',
        't':'ト', 'v':'ヴ', 'w':'ゥ', 'x':'クス', 'y':'ィ', 'z':'ズ',
        }
    

    kana_asist = { 'a':'ァ', 'i':'ィ', 'u':'ゥ', 'e':'ェ', 'o':'ォ', }
    
    
    def __romaji2kana():
        romaji_dict = {}
        for tbl in master, romaji_asist:
            for k, v in tbl.items(): romaji_dict[k] = v
        
        romaji_keys = romaji_dict.keys()
        romaji_keys = sorted(romaji_keys, reverse=True)
        
        re_roma2kana = re.compile("|".join(map(re.escape, romaji_keys)))
        # m の後ろにバ行、パ行のときは "ン" と変換
        rx_mba = re.compile("m(b|p)([aiueo])")
        # 子音が続く時は "ッ" と変換
        rx_xtu = re.compile(r"([bcdfghjklmpqrstvwxyz])\1")
        # 母音が続く時は "ー" と変換
        rx_a__ = re.compile(r"([aiueo])\1")
        
        def _romaji2katakana(text):
            result = text.lower()
            result = rx_mba.sub(r"ン\1\2", result)
            result = rx_xtu.sub(r"ッ\1"  , result)
            result = rx_a__.sub(r"\1ー"  , result)
            return re_roma2kana.sub(lambda x: romaji_dict[x.group(0)], result)
        
        def _romaji2hiragana(text):
            result = _romaji2katakana(text)
            return katakana2hiragana(result)
        
        return _romaji2katakana, _romaji2hiragana
    
    
    def __kana2romaji():
        kana_dict = {}
        for tbl in master, kana_asist:
            for k, v in tbl.items(): kana_dict[v] = k

        kana_keys = kana_dict.keys()
        kana_keys = sorted(kana_keys, reverse=True)
        
        re_kana2roma = re.compile("|".join(map(re.escape, kana_keys)))
        rx_xtu = re.compile("ッ(.)") # 小さい "ッ" は直後の文字を２回に変換
        rx_ltu = re.compile("ッ$"  ) # 最後の小さい "ッ" は消去(?)
        rx_er  = re.compile("(.)ー") # "ー"は直前の文字を２回に変換
        rx_n   = re.compile(r"n(b|p)([aiueo])") # n の後ろが バ行、パ行 なら m に修正
        rx_oo  = re.compile(r"([aiueo])\1")      # oosaka → osaka
        
        def _kana2romaji(text):
            result = hiragana2katakana(text)
            result = re_kana2roma.sub(lambda x: kana_dict[x.group(0)], result)
            result = rx_xtu.sub(r"\1\1" , result)
            result = rx_ltu.sub(r""     , result)
            result = rx_er.sub (r"\1\1" , result)
            result = rx_n.sub  (r"m\1\2", result)
            result = rx_oo.sub (r"\1"   , result)
            return result
        return _kana2romaji
    
    a, b = __romaji2kana()
    c    = __kana2romaji()
    
    return  a, b, c


romaji2katakana, romaji2hiragana, kana2romaji = _make_romaji_convertor()

if __name__=='__main__':
    for s in ("nambda", "maitta", "ping pong"):
        print(s, "\t>\t", romaji2hiragana(s))
    print("=" * 20)
