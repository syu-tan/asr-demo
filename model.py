# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch.nn as nn
import numpy as np


class MyMTLModel(nn.Module):
    ''' Attention RNN + CTC によるEnd-to-Endモデルの定義
    dim_in:            入力次元数
    dim_enc_hid:       エンコーダの隠れ層の次元数
    dim_enc_proj:      エンコーダのProjection層の次元数
                       (これがエンコーダの出力次元数になる)
    dim_dec_hid:       デコーダのRNNの次元数
    dim_out:           出力の次元数(sosとeosを含む全トークン数)
    dim_att:           Attention機構の次元数
    att_filter_size:   LocationAwareAttentionのフィルタサイズ
    att_filter_num:    LocationAwareAttentionのフィルタ数
    sos_id:            <sos>トークンの番号
    enc_bidirectional: Trueにすると，エンコーダに
                       bidirectional RNNを用いる
    enc_sub_sample:    エンコーダにおいてレイヤーごとに設定する，
                       フレームの間引き率
    enc_rnn_type:      エンコーダRNNの種類．'LSTM'か'GRU'を選択する
    '''
    def __init__(self, dim_in, dim_enc_hid, dim_enc_proj, 
                 dim_dec_hid, dim_out, dim_att, 
                 att_filter_size, att_filter_num,
                 sos_id, att_temperature=1.0,
                 enc_num_layers=2, dec_num_layers=2, 
                 enc_bidirectional=True, enc_sub_sample=None, 
                 enc_rnn_type='LSTM'):
        super(MyMTLModel, self).__init__()

        # エンコーダを作成
        self.encoder = Encoder(dim_in=dim_in, 
                               dim_hidden=dim_enc_hid, 
                               dim_proj=dim_enc_proj, 
                               num_layers=enc_num_layers, 
                               bidirectional=enc_bidirectional, 
                               sub_sample=enc_sub_sample, 
                               rnn_type=enc_rnn_type)
        
        # デコーダを作成
        self.decoder = Decoder(dim_in=dim_enc_proj, 
                               dim_hidden=dim_dec_hid, 
                               dim_out=dim_out, 
                               dim_att=dim_att, 
                               att_filter_size=att_filter_size, 
                               att_filter_num=att_filter_num, 
                               sos_id=sos_id, 
                               att_temperature=att_temperature,
                               num_layers=dec_num_layers)

        # CTC出力層
        # 出力次元数はdim_outから<sos>分の1を引いた値
        self.ctc_out = nn.Linear(in_features=dim_enc_proj,
                                 out_features=dim_out)


    def forward(self,
                input_sequence,
                input_lengths,
                label_sequence=None):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
        label_sequence: 各発話の正解ラベル系列(学習時に用いる) [B x Tout]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        # エンコーダに入力する
        enc_out, enc_lengths = self.encoder(input_sequence,
                                            input_lengths)
 
        # CTC出力層にエンコーダ出力を入れる
        ctc_output = self.ctc_out(enc_out)

        # デコーダに入力する
        dec_out = self.decoder(enc_out,
                               enc_lengths,
                               label_sequence)
        # デコーダ出力と，エンコーダ出力、
        # エンコーダ出力系列長情報を出力する
        return dec_out, ctc_output, enc_lengths


    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(utt, filename)

# -*- coding: utf-8 -*-

#
# RNN エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    ''' エンコーダ
    dim_in:        入力特徴量の次元数
    dim_hidden:    隠れ層の次元数(bidirectional=Trueの場合，
                   実際の次元数はdim_hidden * 2)
    dim_proj:      Projection層の次元数
                   (これがエンコーダの出力次元数になる)
    num_layers:    RNN層(およびProjection層)の数
    bidirectional: Trueにすると，bidirectional RNNを用いる
    sub_sample:    レイヤーごとに設定する，フレームの間引き率
                   num_layers=4のとき，sub_sample=[1,2,3,1]とすると
                   2層目でフレーム数を1/2に，3層目で1/3にする
                   (出力のフレーム数は1/6になる)
    rnn_type:      'LSTM'か'GRU'を選択する
    '''
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_proj,
                 num_layers=2, 
                 bidirectional=True,
                 sub_sample=None,
                 rnn_type='LSTM'):
        super(Encoder, self).__init__()
        # RNN層の数
        self.num_layers = num_layers

        # RNN層は1層ずつ定義して，リスト化する
        rnn = []
        for n in range(self.num_layers):
            # RNNへの入力次元数は，
            # 最初の層のみdim_in，それ以外はdim_proj
            input_size = dim_in if n == 0 else dim_proj
            # rnn_type がGRUならGRUを，それ以外ならLSTMを用いる
            if rnn_type == 'GRU':
                rnn.append(nn.GRU(input_size=input_size, 
                                  hidden_size=dim_hidden,
                                  num_layers=1, 
                                  bidirectional=bidirectional, 
                                  batch_first=True))
            else:
                rnn.append(nn.LSTM(input_size=input_size, 
                                   hidden_size=dim_hidden, 
                                   num_layers=1, 
                                   bidirectional=bidirectional, 
                                   batch_first=True))
        # 標準のリスト型のままだと，
        # Pytorchで扱えないので，ModuleListに変換する
        self.rnn = nn.ModuleList(rnn)

        # sub_sample の定義
        if sub_sample is None:
            # 定義されていない場合は，フレームの間引きを行わない
            # (sub_sampleは全要素1のリストにする)
            self.sub_sample = [1 for i in range(num_layers)]
        else:
            # 定義されている場合は，それを用いる
            self.sub_sample = sub_sample

        # Projection層もRNN層と同様に1層ずつ定義する
        proj = []
        for n in range(self.num_layers):
            # Projection層の入力次元数 = RNN層の出力次元数．
            # bidiractional=Trueの場合は次元数が2倍になる
            input_size = dim_hidden * (2 if bidirectional else 1)
            proj.append(nn.Linear(in_features=input_size, 
                                  out_features=dim_proj))
        # RNN層と同様，ModuleListに変換する
        self.proj = nn.ModuleList(proj)
 
        
    def forward(self, sequence, lengths):
        ''' ネットワーク計算(forward処理)の関数
        sequence: 各発話の入力系列 [B x T x D]
        lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B: ミニバッチ内の発話数(ミニバッチサイズ)
          T: 入力テンソルの系列長(ゼロ埋め部分含む)
          D: 入力次元数(dim_in)
        '''
        # 出力とその長さ情報を入力で初期化
        output = sequence
        output_lengths = lengths
 
        # num_layersの数だけ，RNNとProjection層へ交互に入力する
        for n in range(self.num_layers):
            # RNN へ入力するため，
            # 入力をPackedSequenceデータに変換する
            rnn_input \
                = nn.utils.rnn.pack_padded_sequence(output, 
                                                  output_lengths, 
                                                  batch_first=True)

            # GPUとcuDNNを使用している場合，
            # 以下の1行を入れると処理が速くなる
            # (パラメータデータのポインタをリセット)
            self.rnn[n].flatten_parameters()

            # RNN層に入力する
            output, (h, c) = self.rnn[n](rnn_input)

            # RNN出力をProjection層へ入力するため，
            # PackedSequenceデータからtensorへ戻す
            output, output_lengths \
                = nn.utils.rnn.pad_packed_sequence(output, 
                                                  batch_first=True)

            # sub sampling (間引き)の実行
            # この層における間引き率を取得
            sub = self.sub_sample[n]
            if sub > 1:
                # 間引きを実行する
                output = output[:, ::sub]
                # フレーム数を更新する 
                # 更新後のフレーム数=(更新前のフレーム数+1)//sub
                output_lengths = (output_lengths+1) // sub
            
            # Projection層に入力する
            output = self.proj[n](output)

        # sub samplingを実行した場合はフレーム数が変わるため，
        # 出力のフレーム数情報も出力する
        return output, output_lengths

# -*- coding: utf-8 -*-

#
# Attention RNN のデコーダ部の実装です．
# 参考文献
#   - D. Bahdanau, et al., 
#     ``End-to-end attention-based large vocabulary speech 
#       recognition,''
#     in Proc. ICASSP, 2016.
#   - J. Chorowski, et al.,
#     ``Attention-based models for speech recognition,''
#     in Proc. NIPS , 2015.
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt


class Decoder(nn.Module):
    ''' デコーダ
    dim_in:          入力系列(=エンコーダ出力)の次元数
    dim_hidden:      デコーダRNNの次元数
    dim_out:         出力の次元数(sosとeosを含む全トークン数)
    dim_att:         Attention機構の次元数
    att_filter_size: LocationAwareAttentionのフィルタサイズ
    att_filter_num:  LocationAwareAttentionのフィルタ数
    sos_id:          <sos>トークンの番号
    att_temperature: Attentionの温度パラメータ
    num_layers:      デコーダRNNの層の数
    '''
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 dim_att,
                 att_filter_size, 
                 att_filter_num,
                 sos_id,
                 att_temperature=1.0,
                 num_layers=1):

        super(Decoder, self).__init__()

        # <sos>と番号を設定
        self.sos_id = sos_id

        # 入力次元数と出力次元数
        self.dim_in = dim_in
        self.dim_out = dim_out

        # 1ステップ前の出力を入力するembedding層
        # (次元数がdim_outのベクトルから次元数が
        #  dim_hiddenのベクトルに変換する)
        self.embedding = nn.Embedding(dim_out, dim_hidden)
        
        # Location aware attention
        self.attention = LocationAwareAttention(dim_in,
                                                dim_hidden, 
                                                dim_att,
                                                att_filter_size, 
                                                att_filter_num,
                                                att_temperature)

        # RNN層
        # RNNには1ステップ前の出力(Embedding後)と
        # エンコーダ出力(Attention後)が入力される．
        # よってRNNの入力次元数は
        # dim_hidden(Embedding出力の次元数) \
        #   + dim_in(エンコーダ出力の次元数)
        self.rnn = nn.LSTM(input_size=dim_hidden+dim_in, 
                           hidden_size=dim_hidden,
                           num_layers=num_layers, 
                           bidirectional=False,
                           batch_first=True)

        # 出力層
        self.out = nn.Linear(in_features=dim_hidden,
                             out_features=dim_out)

        # Attention重み行列(表示用)
        self.att_matrix = None


    def forward(self, enc_sequence, enc_lengths, label_sequence=None):
        ''' ネットワーク計算(forward処理)の関数
        enc_sequence:   各発話のエンコーダ出力系列
                        [B x Tenc x Denc]
        enc_lengths:    各発話のエンコーダRNN出力の系列長 [B]
        label_sequence: 各発話の正解ラベル系列(学習時に用いる)
                        [B x Tout]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tenc: エンコーダRNN出力の系列長(ゼロ埋め部分含む)
          Denc: エンコーダRNN出力の次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        label_sequenceは，学習時にのみ与える
        '''
        # 入力の情報(バッチサイズ，device(cpu or cuda))を得る
        batch_size = enc_sequence.size()[0]
        device = enc_sequence.device

        #
        # デコーダの最大ステップ数を決める
        #
        if label_sequence is not None:
            # 学習時:
            #   = ラベル情報が与えられている場合は
            #     ラベル系列長を使う
            max_step = label_sequence.size()[1]
        else:
            # 評価時:
            #   = ラベル情報が与えられていない場合は
            #     エンコーダ出力系列長を使う
            max_step = enc_sequence.size()[1]

        #
        # 各内部パラメータの初期化
        #
        # 1ステップ前のトークン．初期値は <sos> とする
        prev_token = torch.ones(batch_size, 1,
                                dtype=torch.long) * self.sos_id
        # デバイス(CPU/GPU)に配置
        prev_token = prev_token.to(device=device,
                                   dtype=torch.long)
        # 1ステップ前のRNN出力とAttention重みをNoneで初期化する
        prev_rnnout = None
        prev_att = None
        # 1ステップ前のRNN内部パラメータ(h, c)もNoneで初期化する
        prev_h_c = None
        # Attentionの内部パラメータをリセットする
        self.attention.reset()

        # 出力テンソルを用意 [batch_size x max_step x dim_out]
        output = torch.zeros(batch_size, max_step, self.dim_out)
        # デバイス(CPU/GPU)に配置
        output = output.to(device=device, dtype=enc_sequence.dtype)

        # 表示用Attention重み行列の初期化
        self.att_matrix = torch.zeros(batch_size, 
                                      max_step,
                                      enc_sequence.size(1))

        #
        # 最大ステップの数だけデコーダを動かす
        #
        for i in range(max_step):
            #
            # 1. Attentionを計算し，コンテキストベクトル
            #    (重みづけ和されたエンコーダ出力)と，
            #    Attention重みを得る
            #
            context, att_weight = self.attention(enc_sequence,
                                                 enc_lengths, 
                                                 prev_rnnout,
                                                 prev_att)
            #
            # 2. RNNを1ステップ分動かす
            #
            # 1ステップ前のトークンをEmbedding層に通す
            prev_token_emb = self.embedding(prev_token)
            # prev_token_embとコンテキストベクトルを結合し，
            # RNNに入力する．RNN入力のテンソルサイズは
            # (batch_size, 系列長(=1), dim_in)なので，
            # contextにviewを用いてサイズを合わせた上で結合する
            context = context.view(batch_size, 1, self.dim_in)
            rnn_input = torch.cat((prev_token_emb, context), dim=2)
            # RNNに通す
            rnnout, h_c = self.rnn(rnn_input, prev_h_c)

            #
            # 3. RNN出力を線形層に通す
            #
            out = self.out(rnnout)
            # 出力テンソルにoutを格納
            output[:,i,:] = out.view(batch_size, self.dim_out)
            
            #
            # 4. 1ステップ前のRNN出力とRNN内部パラメータ，
            #    Attention重み，トークンを更新する．
            #
            prev_rnnout = rnnout
            prev_h_c = h_c
            prev_att = att_weight
            # トークンの更新
            if label_sequence is not None:
                # 学習時:
                #  = 正解ラベルが与えられている場合はそれを用いる
                prev_token = label_sequence[:,i].view(batch_size,1)
            else:
                # 評価時:
                #  = 正解ラベルが与えられていない場合は，
                #    予測値を用いる
                _, prev_token = torch.max(out, 2)

            # 表示用Attention重み行列
            self.att_matrix[:,i,:] = att_weight

        return output

    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        att_mat = self.att_matrix[utt].detach().numpy()
        # プロットの描画領域を作成
        plt.figure(figsize=(5,5))
        # カラーマップのレンジを調整
        att_mat -= np.max(att_mat)
        vmax = np.abs(np.min(att_mat)) * 0.0
        vmin = - np.abs(np.min(att_mat)) * 0.7
        # プロット
        plt.imshow(att_mat, 
                   cmap = 'gray',
                   vmax = vmax,
                   vmin = vmin,
                   aspect = 'auto')
        # 横軸と縦軸のラベルを定義
        plt.xlabel('Encoder index')
        plt.ylabel('Decoder index')

        # プロットを保存する
        plt.savefig(filename)
        plt.close()
        

# -*- coding: utf-8 -*-

#
# Attention (Location aware attention) の実装です．
# 参考文献
#   - D. Bahdanau, et al., 
#     ``End-to-end attention-based large vocabulary speech
#       recognition,''
#     in Proc. ICASSP, 2016.
#   - J. Chorowski, et al.,
#     ``Attention-based models for speech recognition,''
#     in Proc. NIPS , 2015.
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    ''' Location aware attention
    dim_encoder:   エンコーダRNN出力の次元数
    dim_decoder:   デコーダRNN出力の次元数
    dim_attention: Attention機構の次元数
    filter_size:   location filter (前のAttention重みに
                   畳み込まれるフィルタ)のサイズ
    filter_num:    location filterの数
    temperature:   Attention重み計算時に用いる温度パラメータ
    '''
    def __init__(self,
                 dim_encoder,
                 dim_decoder,
                 dim_attention,
                 filter_size, 
                 filter_num,
                 temperature=1.0):

        super(LocationAwareAttention, self).__init__()

        # F: 前のAttention重みに畳み込まれる畳み込み層
        self.loc_conv = nn.Conv1d(in_channels=1,
                                  out_channels=filter_num, 
                                  kernel_size=2*filter_size+1,
                                  stride=1, 
                                  padding=filter_size,
                                  bias=False)
        # 以下三つの層のうち，一つのみbiasをTrueにし，他はFalseにする
        # W: 前ステップのデコーダRNN出力にかかる射影層
        self.dec_proj = nn.Linear(in_features=dim_decoder, 
                                  out_features=dim_attention,
                                  bias=False)
        # V: エンコーダRNN出力にかかる射影層
        self.enc_proj = nn.Linear(in_features=dim_encoder, 
                                  out_features=dim_attention,
                                  bias=False)
        # U: 畳み込み後のAttention重みにかかる射影層
        self.att_proj = nn.Linear(in_features=filter_num, 
                                  out_features=dim_attention,
                                  bias=True)
        # w: Ws + Vh + Uf + b にかかる線形層
        self.out = nn.Linear(in_features=dim_attention,
                             out_features=1)

        # 各次元数
        self.dim_encoder = dim_encoder
        self.dim_decoder = dim_decoder
        self.dim_attention = dim_attention

        # 温度パラメータ
        self.temperature = temperature

        # エンコーダRNN出力(h)とその射影(Vh)
        # これらは毎デコードステップで同じ値のため，
        # 一回のみ計算し，計算結果を保持しておく
        self.input_enc = None
        self.projected_enc = None
        # エンコーダRNN出力の，発話ごとの系列長
        self.enc_lengths = None
        # エンコーダRNN出力の最大系列長
        # (=ゼロ詰めしたエンコーダRNN出力の系列長)
        self.max_enc_length = None
        # Attentionマスク
        # エンコーダの系列長以降
        # (ゼロ詰めされている部分)の重みをゼロにするマスク
        self.mask = None


    def reset(self):
        ''' 内部パラメータのリセット
            この関数は1バッチの処理を行うたびに，
            最初に呼び出す必要がある
        '''
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None

    
    def forward(self, 
                input_enc,
                enc_lengths,
                input_dec=None,
                prev_att=None):
        ''' ネットワーク計算(forward処理)の関数
        input_enc:   エンコーダRNNの出力 [B x Tenc x Denc]
        enc_lengths: バッチ内の各発話のエンコーダRNN出力の系列長 [B]
        input_dec:   前ステップにおけるデコーダRNNの出力 [B x Ddec]
        prev_att:    前ステップにおけるAttention重み [B x Tenc]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tenc: エンコーダRNN出力の系列長(ゼロ埋め部分含む)
          Denc: エンコーダRNN出力の次元数(dim_encoder)
          Ddec: デコーダRNN出力の次元数(dim_decoder)
        '''
        # バッチサイズ(発話数)を得る
        batch_size = input_enc.size()[0]

        #
        # エンコーダRNN出力とその射影ベクトルを一度のみ計算
        #
        if self.input_enc is None:
            # エンコーダRNN出力(h)
            self.input_enc = input_enc
            # 各発話の系列長
            self.enc_lengths = enc_lengths
            # 最大系列長
            self.max_enc_length = input_enc.size()[1]
            # 射影を行う(Vhの計算)
            self.projected_enc = self.enc_proj(self.input_enc)
        
        #
        # 前ステップにおけるデコーダRNN出力を射影する(Wsの計算)
        #
        # 前のデコーダRNN出力が無い場合は初期値としてゼロ行列を使用
        if input_dec is None:
            input_dec = torch.zeros(batch_size, self.dim_decoder)
            # 作成したテンソルをエンコーダRNN出力と
            # 同じデバイス(GPU/CPU)に配置
            input_dec = input_dec.to(device=self.input_enc.device, 
                                     dtype=self.input_enc.dtype)
        # 前のデコーダRNN出力を射影する
        projected_dec = self.dec_proj(input_dec)

        #
        # 前ステップにおけるのAttention重み情報を
        # 射影する(Uf+bの計算)
        #
        # Attentionマスクを作成
        if self.mask is None:
            self.mask = torch.zeros(batch_size, 
                                    self.max_enc_length, 
                                    dtype=torch.bool)
            # バッチ内の各発話について，その発話の
            # 系列長以上の要素(つまりゼロ埋めされている部分)を
            # 1(=マスキング対象)にする
            for i, length in enumerate(self.enc_lengths):
                length = length.item()
                self.mask[i, length:] = 1
            # 作成したテンソルをエンコーダRNN出力と
            # 同じデバイス(GPU/CPU)に配置
            self.mask = self.mask.to(device=self.input_enc.device)

        # 前のAttention重みが無い場合は初期値として，
        # 一様の重みを与える
        if prev_att is None:
            # 全ての要素を1のテンソルを作成
            prev_att = torch.ones(batch_size, self.max_enc_length)
            # 発話毎の系列長で割る
            # このとき，prev_attは2次のテンソル，
            # enc_lengthsは1次のテンソルなので，
            # view(batch_size, 1)によりenc_lengthsを
            # 2次テンソルの形にしてから割り算する
            prev_att = prev_att \
                     / self.enc_lengths.view(batch_size, 1)
            # 作成したテンソルをエンコーダRNN出力と
            # 同じデバイス(GPU/CPU)に配置
            prev_att = prev_att.to(device=self.input_enc.device, 
                                   dtype=self.input_enc.dtype)
            # 発話長以降の重みをゼロにするようマスキングを実行
            prev_att.masked_fill_(self.mask, 0)

        # Attention重みの畳み込みを計算する {f} = F*a
        # このとき，Conv1Dが受け付ける入力のサイズは
        # (batch_size, in_channels, self.max_enc_length)
        # (in_channelsは入力のチャネル数で，
        # このプログラムではin_channels=1) 
        # サイズを合わせるため，viewを行う
        convolved_att \
            = self.loc_conv(prev_att.view(batch_size, 
                                          1, self.max_enc_length))
 
        # convolved_attのサイズは
        # (batch_size, filter_num, self.max_enc_length)
        # Linearレイヤーが受け付ける入力のサイズは
        # (batch_size, self.max_enc_length, filter_num) なので，
        # transposeにより1次元目と2次元目をの入れ替えた上で
        # att_projに通す
        projected_att = self.att_proj(convolved_att.transpose(1, 2))
        
        #
        # Attention重みを計算する
        # 
        # この時点での各テンソルのサイズは
        # self.projected_enc: (batch_size, self.max_enc_length, 
        #                      self.dim_attention)
        # projected_dec: (batch_size, self.dim_attention)
        # projected_att: (batch_size, self.max_enc_length, self.dim_attention)
        # projected_decのテンソルの次元数を合わせるため，viewを用いる
        projected_dec = projected_dec.view(batch_size,
                                           1,
                                           self.dim_attention)

        # scoreを計算するため，各射影テンソルの加算，
        # tanh，さらに射影を実施
        # w tanh(Ws + Vh + Uf + b)
        score = self.out(torch.tanh(projected_dec \
                                    + self.projected_enc 
                                    + projected_att))

        # 現時点のscoreのテンソルサイズは
        # (batch_size, self.max_enc_length, 1)
        # viewを用いて元々のattentionのサイズに戻す
        score = score.view(batch_size, self.max_enc_length)

        # マスキングを行う
        # (エンコーダRNN出力でゼロ埋めされている部分の
        # 重みをゼロにする)
        # ただし，この後softmax関数の中で計算される
        # exp(score)がゼロになるように
        # しないといけないので，scoreの段階では0ではなく，
        # 0の対数値である-infで埋めておく
        score.masked_fill_(self.mask, -float('inf'))

        # 温度付きSoftmaxを計算することで，Attention重みを得る
        att_weight = F.softmax(self.temperature * score, dim=1)

        # att_weightを使って，エンコーダRNN出力の重みづけ和を計算し，
        # contextベクトルを得る
        # (viewによりinput_encとattention_weightの
        # テンソルサイズを合わせている)
        context \
            = torch.sum(self.input_enc * \
                att_weight.view(batch_size, self.max_enc_length, 1),
                dim=1)

        # contextベクトルとattention重みを出力
        return context, att_weight

