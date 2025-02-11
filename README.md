概要

本プロジェクトでは、OpenPoseを用いた姿勢推定を活用し、入力動画を利用してパンチ動作を検出するシステムを構築した。姿勢データのクラスタリングとkNN分類を組み合わせることで、パンチ動作の認識を行う。

使用技術

2.1 OpenPose

OpenPoseは、人体の関節点（キーポイント）をリアルタイムで検出するためのライブラリであり、本プロジェクトでは手の動きを中心に解析を行った。

2.2 kNN

kNNは、学習済みのデータポイントに基づいて未知のデータを分類する機械学習アルゴリズムである。本プロジェクトでは、過去の姿勢データをもとにパンチの動きを分類するのに使用した。

2.3 クラスタリング

パンチ動作の特徴を抽出するため、クラスタリングを用いてデータをグループ化し、パンチ特有のパターンを見出すことを試みた。





3. パンチがカウントされる条件

パンチとしてカウントされるのは以下の条件を満たす場合である。

手の速度が一定の閾値を超えている。

肘の角度が適切に変化し、パンチのフォームに合致している。

kNN分類により、事前に定義されたパンチのクラスタに属する。

4. 改善点

精度向上のための学習データ拡充: より多くの異なるパンチのデータを収集し、分類モデルの精度を向上させる。

異なるパンチ種類の識別: ジャブ、ストレート、フックなど異なるパンチの動作を識別できるようにする。

リアルタイム処理の高速化: GPU処理の最適化やモデルの軽量化を行い、フレームレートを向上させる。

誤検出の削減: 手以外の動き（例えば歩行や腕の振り）をパンチと誤認しないよう、フィルタリング手法を強化する。

5. まとめ

本システムはOpenPoseによる姿勢推定、クラスタリング、kNN分類を組み合わせることで、リアルタイムでパンチ動作を認識することを可能にした。今後の課題としては、パンチ種類の識別や誤検出の削減が挙げられる。精度の向上と処理速度の最適化を進めることで、スポーツトレーニングやVRゲームなど幅広い応用が期待できる。


