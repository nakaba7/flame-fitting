実行の順番：
1. ラズパイでCommandCapture_a(or b).pyを実行後，CaptureChessBoard_Raspi.pyを実行．その後，ラズパイから画像をコピーする．
//2. imagerotate.pyで画像を回転(aは-90, bは90). ← ラズパイでCommandCapture_x.pyへ組み込み済
3. CameraCalibration.pyをChessBoard_aとChessBoard_bに対して実行．
4. StereoCalibration.pyを実行．
5. FaceLandmark_manual.pyで顔画像のアノテーション
6. Point2D_2_3D.pyでアノテーションした特徴点を3次元座標へ変換