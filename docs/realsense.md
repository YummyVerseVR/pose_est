# RealSense D435i RGB-D 食事姿勢サポート

## 状態と対象

2026-11-12実装。Windows 11 x64 + USB 3.x + D435iを主対象とする。
合成深度・モックSDK・macOS上のWebcam用CLIで検証。Windows実行、実際の
D435i入力、精度、食事中の使用性は未検証。LinuxはSDK/udev設定が必要。
macOSのRealSense実行は保証しない。Kinect v2とIMU補正は対象外。

研究用の着座姿勢への気づき支援であり、誤嚥・窒息・嚥下安全性を判定しない。
基準は専門職と確認した本人の無理のない座位。顎引き等を一律に指示しない。
角度差・2.5秒の持続・10/35秒の通知間隔・深度品質閾値は工学/UXの仮置きで、
医学的安全値ではない。個別の指示は設定ファイルのmetricsで指定する。
`support_profile: clinician_defined` はその設定の由来を識別するだけで、
自動で治療姿勢や閾値を選ばない。飲水・咀嚼・嚥下を自動識別しない。Pで停止する。

## Windows 11 セットアップ（PowerShell、実機手順は未実行）

1. Python 3.12 x64とuv、公式RealSense SDK/Viewerを導入。D435iをUSB 3.xへ直接接続。
   Viewerでカラーと深度を確認し、Viewerを閉じてカメラをアプリへ解放する。
2. リポジトリ内で次を実行する。

```powershell
uv sync --python 3.12 --extra realsense
uv run python scripts/download_model.py
Copy-Item config.realsense.example.json config.json
uv run pose-est --help
uv run pose-est --config config.json --camera-source realsense
# SDK不要の従来モード
uv sync --python 3.12
uv run pose-est --config config.example.json --camera-source webcam
```

2026-11-12の[PyPI metadata](https://pypi.org/pypi/pyrealsense2/json)を取得し、
`pyrealsense2 2.59.1.11053` の `cp312-cp312-win_amd64.whl` を確認。
cp313/314 Windows x64、cp312 Linux x86_64/aarch64 wheelも存在した。
wheelの存在確認はWindowsへのインストール・実行検証を意味しない。
推奨の再現環境はPython 3.12 x64。arm64 Windowsや将来のPythonへ一般化しない。
optional extraのみSDKを追加し、Webcam/CLIでSDKをimportしない。

SDK不足、未接続、複数台（sensor_serial未指定）は案内付きで終了。
取得はframe_timeout_ms（既定3000ms）の有限待ち。フレーム欠落/USB断は
判定を続けず終了・解放する。Q/Esc/Ctrl+C、初期化/推論例外時も解放する。

## 配置と再登録

正面0°に対して側方75°、距離1.3m（開始範囲1.2〜1.5m）、床からレンズ1.0m
（0.9〜1.1m）、水平固定で開始。鼻・耳・肩・股関節付近をRGB画面に含め、
机・食器・腕・椅子・介助者の遮蔽を減らす。75°は臨床由来の推奨角ではない。
光学的な設置側camera_sideと本人の解剖学的左右は別である。

Cで基準を採取。D435i設定例は30秒・各指標300サンプル（既存Webcamは3秒/30件）。
画面のSamplesに指標別件数を表示。設定で有効にした全指標が十分に観測されるまで
完了しない。頭しか映らない場合に体幹基準が登録された扱いにはしない。
基準登録できない場合は配置を直すか、個別設定で観測できない指標を無効にする。

カメラの移動・傾き変更時は必ずCで再登録し、placement_idも変える。
水平/重力の自動検証、移動検知はない。設置メタデータは手動記録である。
sensor_serial、実intrinsics、depth scale、解像度/fps、配置、品質/指標設定が異なる
version 2基準は拒否し、再登録を案内する。基準を無効化してCを押すと旧基準を
継続利用しない。Webcam v1基準はWebcamのみ後方互換で読める。

## 計測と品質

FrameSource → 同期frameset → colorへalignしたZ16とBGR → image landmarks →
robust depth sampling → SDK deprojection → RGB-D observation → 既存Analyzer。
MediaPipeのworld_landmarksはRGB-D計測へ使わない。

実解像度で正規化座標を画素化。visibility/presence、範囲、有限性を検査し、
5×5パッチの0・非有限・範囲外深度を除外する。最低9画素、60%有効が必要。
中央値とMAD（上限0.02m）、中央値±0.06mのinlier率80%で背景混合を抑制。
confidence = landmark confidence × 有効率 × inlier率 × (1 − MAD/上限)。
最低0.5未満は数値を出さずunknown。これらも未検証の工学設定で、
平坦な背景だけが映る誤対応を完全には排除できない。

カメラ座標はm、+X画面右、+Y下、+Zカメラから遠ざかる方向。
実運用はaligned depth intrinsicsとrs2_deproject_pixel_to_pointを使用し歪みもSDKへ渡す。
テストのみ無歪みpinholeを使用。表示鏡像は計測値と左右定義を変更しない。

| 指標 | 定義・符号・限界 |
| --- | --- |
| head_pitch_deg | 選択した耳→鼻の水平成分に対するピッチ。上向き正。頸椎角や顎引きの直接測定ではない |
| torso_forward_deg | 同側の股関節→肩を、基準時の水平前方軸へ投影し、鉛直上方向とのatan2。前傾正。脊柱後弯は測らない |
| head_forward_ratio | 耳−肩の基準前方成分/現在の体幹長。身体全体の平行移動は相殺 |
| hip_position_delta | 基準採取中の最初の有効な股関節XYZからのユークリッド距離m。実験的・既定無効。骨盤後傾や滑り座りは確定しない |

基準登録時に、有効な指標に必要な耳・肩・腰が観測できる側から最低信頼度が高い側を選択し、その解剖学的側を固定する。
最初の有効な耳→鼻の水平ベクトルを前方軸として保存。顔を後から回しても
体幹の前方軸は変えない。基準採取中は本人が確認した座位・向きを保つ。
各指標は必要なランドマークだけを要求し、鼻欠損でも体幹、腰欠損でも頭部を観測可能。
欠損で古い警告を表示/通知せず、その項目の平滑化と警告継続時間をリセットする。
復帰後は新たに持続条件を満たす必要がある。

横傾き、回旋、足底接地は判定しない。RGB-Dでfeet.enabled=trueは設定エラー。
unknownはokではない。UIはモード、設置側、RGB-D有効率（鼻/選択側耳肩腰の
パッチ有効率の平均、推論精度ではない）、各指標の追跡/unknown原因、基準、停止状態を表示。

## 反応を早める設定例

`config.realsense-responsive.example.json` は、実機使用後の要望を受けた試用設定。
注意持続1.2秒、復帰0.5秒、平滑化alpha=0.5、基準採取最短8秒/各指標60件。
時間とサンプル数の両方が必要なので、遮蔽や低fpsでは8秒を超える。
角度差・深度品質の条件と通知間隔10/35秒は通常設定と同じ。
実機起動確認用にフレーム取得タイムアウトは通常の3秒から15秒へ延長している。
フレームが届かない際はエラー終了まで最大15秒待つ。姿勢の注意持続1.2秒とは別の設定。
短い食事動作で通知が増える可能性があり、個人別の使用性評価が必要。

```bash
uv run pose-est --config config.realsense-responsive.example.json --camera-source realsense
```

設定は起動時に読むため再起動が必要。指標設定が変わると旧基準を拒否するのでCで再登録。
合成30fps相当では持続した体幹変化への初回注意が約2.6→1.3秒となった。
これは判定器の試験で、カメラから画面までの実測遅延ではない。
Mac実機での短時間RGB-D取得と未確認項目は[実機記録](d435i-macos-runtime.md)を参照。

## プライバシー

既定では映像・深度・研究ログを保存/送信しない。基準は個人データとしてローカルに
保存される。baseline_realsense*.json等はGit除外。任意名の基準ファイルも共有しない。
研究ログはresearch_log_pathを明示した場合だけJSONL追記する。時刻・指標・信頼度・
状態・通知と品質理由を保存し、画像、深度配列、landmark XYZ、sensor serialは保存しない。
明示する保存先は管理されたローカル領域（例research-logs/session.jsonl、親directoryを
事前作成）とし、対象者の同意・仮名化・アクセス権・保管期間/削除担当を運用側で決める。
個人名や実IDをファイル名へ入れない。削除はログと個人基準の両方を対象にする。
ソフトは同意取得や保存期限を自動管理しない。

## 実機受け入れ（全項目未実施）

- Windows 11 ViewerでRGB/深度、アプリで640×480×30fpsとaligned depth対応を確認。
- 1.3m/75°で鼻耳肩腰の有効率、30〜60秒の基準、持続前傾/後傾/見上げと通知間隔。
- 短いリーチ、咀嚼/飲水模擬で過剰通知がないか、P停止/再開。
- 机・腕・食器遮蔽でunknownとなり虚偽のok/古い通知が出ないか。
- USB抜去、再接続、複数台、Q/Ctrl+Cの解放と有限待ち。

比較は0/45/75/90°、基準座位・持続前傾・後傾・頭部後屈・リーチ・咀嚼/飲水模擬・遮蔽。
手動/基準器との角度誤差、観測率、unknown時間率、検知遅延、誤通知/分、介助負担を記録。
これは工学評価で、誤嚥予防効果の実証ではない。嚥下障害のある参加者を対象にする
研究は別途倫理審査と医療専門職の関与を検討する。

## 原典

- [公式align例](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)
- [公式Projection/Deprojection](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
- [D435i製品情報](https://www.realsenseai.com/products/depth-camera-d435i/)
- 臨床上の限界・既存根拠は[evidence_and_validation](evidence_and_validation.md)。
