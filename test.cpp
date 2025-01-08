#include <opencv2\opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <gflags/gflags.h>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <time.h>
#include <stdio.h>
#include <chrono>



using namespace op;
using namespace cv;
using namespace std;
using namespace rs2;

// OpenPose
DEFINE_string(model_pose, "BODY_25", "Model to be used. E.g. COCO (18 keypoints), MPI (15 keypoints, ~10% faster), "
	"MPI_4_layers (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, "256x256", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
	" decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
	" closest aspect ratio possible to the images or videos to be processed. Using -1 in"
	" any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
	" input value. E.g. the default -1x368 is equivalent to 656x368 in 16:9 resolutions,"
	" e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
	" input image resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
	" If you want to change the initial scale, you actually want to multiply the"
	" net_resolution by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
	" background, instead of being rendered into the original image. Related: part_to_show,"
	" alpha_pose, and alpha_pose.");
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
	" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
	" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
	" more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
	" hide it. Only valid for GPU rendering.");

#define IMAGE_WIDTH 640        // RGB-Dカメラのフレーム画像の幅
#define IMAGE_HEIGHT 360       // RGB-Dカメラのフレーム画像の高さ
#define NUMBER_OF_JOINTS 25    // 関節数
#define MINIMUM_CONFIDENCE 0.1 // 信頼度の許容最小値
#define KNN_NUMBER 3           // kNN法の最近傍数

// 関数プロトタイプ宣言
int clusteringPostures(Mat posturesForClustering, int numberOfClusters, Mat*clusteredLabels);
void drawJoint(int i, int j, float c, float x, float y, Mat plane);
void drawLimb(Mat postures, int sample, int person, Mat image);
void removeWindowsForClusteredCenters(int number_of_clusters);

// 人物表示色の設定
Scalar color[7] =
{ Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),
  Scalar(0,255,255),Scalar(255,0,255),Scalar(255,255,0),Scalar(255,255,255) };

// 各体節の関節対
int limb[15][2] =
{ {1,2},{2,3},{3,4},{1,5},{5,6},{6,7},{1,8},{8,9},
  {8,12},{9,10},{10,11},{12,13},{13,14},{11,22},{14,20} };
#include <map> // クラスタIDに対応するパンチ名を管理

// パンチ名とそのカウント
std::map<int, std::string> clusterToPunchType;
std::map<std::string, int> punchCounts;


// クラスタ名の初期化関数



int main(int argc, char *argv[])
{
	// --------------ビデオ読み込み----------------------------------
	int flag = 0;
	cv::VideoCapture video("sample.mp4");
	cv::VideoCapture video2("test.mp4");
	if (!video.isOpened()) { // エラー処理
		std::cout << "video.error" << std::endl;
		return -1;
	}
	if (!video2.isOpened()) { // エラー処理
		std::cout << "video.error" << std::endl;
		return -1;
	}
	Mat colorImage;

	cv::Mat frame, gray, canny; // フレーム格納用

	int width = video.get(cv::CAP_PROP_FRAME_WIDTH); // 動画から幅を取得
	int height = video.get(cv::CAP_PROP_FRAME_HEIGHT); // 動画から高さを取得
	int count = 0; //パンチの数を数える
	int presponse = -1; //前回のresponseの値
	int presponse2 = -2;//前々回のresponceの値
	float prevLeftHandX = 0, prevLeftHandY = 0; // 左手の前回位置
	float prevRightHandX = 0, prevRightHandY = 0; // 右手の前回位置
	const float punchThreshold = 30.0; // パンチ検出の移動量しきい値

	// <<<---------- RGB - Dカメラを使う場合の初期化処理ここから ----------

	// 距離画像をカラー画像に合わせるためのパラメータ設定
	//rs2::align alignToColor(RS2_STREAM_COLOR);

	// カラー画像と距離画像の映像ストリームの階調数とフレームレート設定
	//config streamConfig;
	/*streamConfig.enable_stream(RS2_STREAM_COLOR, IMAGE_WIDTH, IMAGE_HEIGHT, RS2_FORMAT_BGR8, 30);
	streamConfig.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_Z16, 30);
*/
// 映像ストリーミング開始
///pipeline pipeLine;
//auto profile = pipeLine.start(streamConfig);

// ---------- RGB - Dカメラを使う場合の初期化処理ここまで ---------->>>

// <<<---------- 通常カメラを使う場合の初期化処理ここから ----------

//VideoCapture camera;
//camera.open(0); // カメラ番号は適当なものをセット

// ---------- 通常カメラを使う場合の初期化処理ここまで ---------->>>

// <<<---------- OpenPoseを使うための初期化処理ここから ----------

	const auto outputSize = flagsToPoint(FLAGS_output_resolution, "-1x-1");
	const auto netInputSize = flagsToPoint(FLAGS_net_resolution, "-1x368");
	const auto poseModel = flagsToPoseModel(FLAGS_model_pose);

	if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)	if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
		op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
			__LINE__, __FUNCTION__, __FILE__);

	ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
	CvMatToOpInput cvMatToOpInput{ poseModel };
	CvMatToOpOutput cvMatToOpOutput;
	PoseExtractorCaffe poseExtractorCaffe{ poseModel, FLAGS_model_folder, FLAGS_num_gpu_start };
	poseExtractorCaffe.initializationOnThread();

	// ---------- OpenPoseを使うための初期化処理ここまで ---------->>>

	// <<<---------- クラスタリングのための初期化処理ここから ----------

	// クラスタリング対象姿勢の保存用配列の宣言（最初は暫定的な１サンプルのみ保存可能なサイズ <-- Mat型のサイズ記述方法に注意：Mat (int rows, int cols, int type) ）
	Mat posturesForClustering = Mat(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);
	Mat posturesForClustering2 = Mat(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);

	int storedSamples = 0;	// 格納済データ数（最初は0）
	int save = -1;			// データ保存の有無（最初は保存なしモード）

	// クラスタ数指定用トラックバーの設置
	namedWindow("Color Image");
	int number_of_clusters = 0; // 変更可能なクラスタ数
	int clusters = 3;
	createTrackbar("Clusters", "Color Image", &clusters, 10);

	// ---------- クラスタリングのための初期化処理ここまで ---------->>>

	// <<<---------- kNN法のための初期化処理ここから ----------

	Ptr<ml::KNearest> knn = ml::KNearest::create();				// kNN識別器の生成
	knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);	// 近傍探索法の設定
	knn->setDefaultK(KNN_NUMBER);								// 近傍数の設定
	knn->setIsClassifier(true);									// 識別結果はカテゴリ
	int recognize = -1;	
	auto lastIncrementTime = std::chrono::steady_clock::now();
	const int delayInMillis = 150; // インクリメント間の待機時間（ミリ秒）// kNNによる認識の有無（最初は認識なしモード）

	// ---------- kNN法のための初期化処理ここまで ---------->>>

	while (1) {
		

		if (flag == 0)
		{
			video >> colorImage;
		}

		if (flag == 0 && colorImage.empty()) {
			flag = 1;
			if (number_of_clusters > 0) removeWindowsForClusteredCenters(number_of_clusters);

			number_of_clusters = clusters;

			Mat clusteredLabels;  // クラスタリング用の各サンプルのクラス格納用配列の宣言

			int result_of_clustering = clusteringPostures(posturesForClustering, number_of_clusters, &clusteredLabels);  // クラスタリング処理(クラスタリング結果受け取り)
			save = -1;  // サンプル保存は中止

			if (result_of_clustering == 0) {
				knn->train(posturesForClustering, ml::ROW_SAMPLE, clusteredLabels);  // kNN法のサンプルの学習
				recognize = 1; // 認識モードON
				
			}
			else recognize = -1;
		}
		if (flag == 1) { video2 >> colorImage; }
		if (flag == 1 && colorImage.empty()) { break; }
		// <<<---------- RGB-Dカメラを使う場合のフレーム画像取得処理ここから ----------

		// RGB-Dカメラを使う場合のカラー画像と距離画像のフレームセットの取得
		//frameset frameSet = video.wait_for_frames();

		//// 距離画像をカラー画像に合わせる
		//frameSet = alignToColor.process(frameSet);

		//// フレームセットからカラー画像と距離画像のフレームを獲得
		//video_frame colorFrame = frameSet.get_color_frame();
		//depth_frame depthFrame = frameSet.get_depth_frame();

		//// Mat型のカラー画像と距離画像データを獲得
		//Mat colorImage(Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, (void*)colorFrame.get_data(), Mat::AUTO_STEP);
		//Mat depthImage16(Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_16UC1, (void*)depthFrame.get_data(), Mat::AUTO_STEP);

		// ---------- RGB-Dカメラを使う場合のフレーム画像取得処理ここまで ---------->>>

		// <<<---------- 通常のカラーカメラを使う場合のフレーム画像の取得ここから ----------

		//Mat colorImage;
		//camera >> colorImage;
		//resize(colorImage, colorImage, Size(IMAGE_WIDTH, IMAGE_HEIGHT));

		// ---------- 通常のカラーカメラを使う場合のフレーム画像の取得ここまで ---------->>>


		// 姿勢データ・クラスタラベル格納用配列への行追加
		if (posturesForClustering.rows <= storedSamples)posturesForClustering.resize(storedSamples + 1);

		// <<<---------- OpenPoseによる姿勢推定処理ここから ----------

		const op::Point<int> imageSize{ colorImage.cols, colorImage.rows };
		vector<double> scaleInputToNetInputs;
		vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;
		tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor.extract(imageSize);
		const auto netInputArray = cvMatToOpInput.createArray(colorImage, scaleInputToNetInputs, netInputSizes);
		poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);

		// ---------- OpenPoseによる姿勢推定処理ここまで ---------->>>


		// <<<---------- 推定結果の姿勢表示処理ここから ----------

		const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
		const auto numberPeopleDetected = poseKeypoints.getSize(0);
		const auto numberBodyParts = poseKeypoints.getSize(1);

		// 人物毎の処理
		for (int i = 0; i < numberPeopleDetected; i++) {

			// 対象人物の推定姿勢データ格納用配列の宣言
			Mat observedPosture(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);
			Mat observedPosture2(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);
			float leftHandX = poseKeypoints[{i, 7, 0}] - poseKeypoints[{i, 1, 0}]; // 左手首
			float leftHandY = poseKeypoints[{i, 7, 1}] - poseKeypoints[{i, 1, 1}];
			float rightHandX = poseKeypoints[{i, 4, 0}] - poseKeypoints[{i, 1, 0}]; // 右手首
			float rightHandY = poseKeypoints[{i, 4, 1}] - poseKeypoints[{i, 1, 1}];
			
			float leftMovement = sqrt(pow(leftHandX - prevLeftHandX, 2) + pow(leftHandY - prevLeftHandY, 2));
			float rightMovement = sqrt(pow(rightHandX - prevRightHandX, 2) + pow(rightHandY - prevRightHandY, 2));
			auto currentTime = std::chrono::steady_clock::now();
			// 前回のインクリメントからの経過時間を計算
			auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastIncrementTime).count();

			// 関節毎の処理
			for (int j = 0; j < numberBodyParts; j++) {

				// i番目の人物のj番目の関節位置(x,y)と信頼度cの取得

				const auto x = poseKeypoints[{i, j, 0}] - poseKeypoints[{i, 8, 0}];
				const auto y = poseKeypoints[{i, j, 1}] - poseKeypoints[{i, 8, 1}];
				const auto c = poseKeypoints[{i, j, 2}];



				// i番目の人物のj番目の関節位置(x,y)を，信頼度cに比例した直径の円でcolorImage上に描画
				drawJoint(i, j, c, x + poseKeypoints[{i, 8, 0}], y + poseKeypoints[{i, 8, 1}], colorImage);

				// 上の関節位置と信頼度を推定姿勢データ格納用配列に格納


				observedPosture.at<float>(3 * j) = x;
				observedPosture.at<float>(3 * j + 1) = y;
				observedPosture.at<float>(3 * j + 2) = c;
				observedPosture2.at<float>(3 * j) = x + poseKeypoints[{i, 8, 0}];
				observedPosture2.at<float>(3 * j + 1) = y + poseKeypoints[{i, 8, 1}];
				observedPosture2.at<float>(3 * j + 2) = c;


				// 0番目の人物の場合
				if (i == 0 && flag == 0) {
					// 上の関節位置と信頼度をクラスタリング対象姿勢の保存用配列にも保存
					posturesForClustering.at<float>(storedSamples, 3 * j) = x;
					posturesForClustering.at<float>(storedSamples, 3 * j + 1) = y;
					posturesForClustering.at<float>(storedSamples, 3 * j + 2) = c;

				}

			}

			// 推定姿勢データ格納用配列を使って体節線の描画
			drawLimb(observedPosture2, 0, i, colorImage);

			// 保存モードがONで0番目の人物の場合はクラスタリング対象姿勢の保存用配列に保存された姿勢サンプル数のカウントアップ
			if (save > 0 && i == 0 && flag == 0) {
				storedSamples++;
				printf("%d samples have stored\n", storedSamples);
			}

			// 認識モードON かつ 0番目の人物の場合
			if (i == 0 && recognize > 0 && flag == 1) {

				// 推定姿勢データ格納用配列を利用して認識・表示

				int response = (int)knn->predict(observedPosture);

				
				
				
				

				if ((response != presponse) && (presponse != presponse2) && (leftMovement > punchThreshold || rightMovement > punchThreshold) &&
					(elapsedTime > delayInMillis)) {
					

					
						count++;
						lastIncrementTime = currentTime; // インクリメントした時点の時刻を記録
						presponse2 = presponse;
						presponse = response;
						
					
				}
				prevLeftHandX = leftHandX;
				prevLeftHandY = leftHandY;
				prevRightHandX = rightHandX;
				prevRightHandY = rightHandY;
				
				

				putText(colorImage, to_string(response), cv::Point(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 3.0, color[response % 7], 3, 8, false);
				putText(colorImage, "Punch Count: "+to_string(count), cv::Point(width -700, 50), FONT_HERSHEY_COMPLEX_SMALL, 3.0, 0, 3, 8, false);
				putText(colorImage, to_string(elapsedTime), cv::Point(10, 150), FONT_HERSHEY_COMPLEX_SMALL, 3.0, 0, 3, 8, false);
			}

		}

		// ---------- 推定結果の姿勢表示処理ここまで ---------->>>

		imshow("Color Image", colorImage);


		int key = cv::waitKey(1);

		// qキーで終了
		if (key == 'q') break;

		// sキーでクラスタリング用のサンプル保存ON/OFF
		else if (key == 's') save *= -1;

		// cキーでクラスタリング


		// rキーでリセット
		else if (key == 'r') {

			// クラスタリング用の保存サンプル初期化
			storedSamples = 0;
			posturesForClustering.resize(1);

			// クラスタリング結果の消去
			if (number_of_clusters > 0) removeWindowsForClusteredCenters(number_of_clusters);

			save = -1;  // サンプル保存は中止
			recognize = -1; // 認識モードOFF

		}

	}

	return 0;

}

// クラスタリング
int clusteringPostures(Mat posturesForClustering, int numberOfClusters, Mat *labels)
{
	// クラスタ数が0のときは何もしない
	if (numberOfClusters == 0) return -1;

	// <<<---------- クラスタリング処理ここから ----------

	// 分類クラスタ，クラスタ中心の出力用配列の宣言
	Mat centers;

	// k平均法の実行
	kmeans(posturesForClustering, numberOfClusters, *labels,
		TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 1, KMEANS_PP_CENTERS, centers);

	for (int k = 0; k < (*labels).rows; k++) {
		int cluster = (*labels).at<int>(k);
		printf("Sample_%d = Cluster_%d\n", k, cluster);
	}

	// ---------- クラスタリング処理ここまで ---------->>>

	// <<<---------- クラスタ中心の描画ここから ----------

	// クラスタ数だけ繰り返し
	for (int i = 0; i < numberOfClusters; i++) {

		// 描画用画像を黒に塗りつぶす
		Mat centroidImage = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);

		// 関節数だけ繰り返し
		for (int j = 0; j < NUMBER_OF_JOINTS; j++) {

			// 関節位置の獲得
			float x, y, c;
			x = centers.at<float>(i, 3 * j);
			y = centers.at<float>(i, 3 * j + 1);
			c = centers.at<float>(i, 3 * j + 2);
			drawJoint(i, j, c, x + 300, y + 300, centroidImage);
		}

		drawLimb(centers + 300, i, i, centroidImage);

		// 表示用ウィンドウ名の設定
		char window_name[20];
		sprintf_s(window_name, 20, "Cluster%d", i);

		// ウィンドウに表示
		imshow(window_name, centroidImage);

	}

	// ---------- クラスタ中心の描画ここまで ---------->>>

	return 0;

}

// 任意の関節位置の描画
void drawJoint(int i, int j, float c, float x, float y, Mat plane)
{

	cv::Point p = cv::Point(x, y);
	circle(plane, p, 5 * c, color[i % 7], -1, 8, 0);
	putText(plane, to_string(j), p, FONT_HERSHEY_COMPLEX_SMALL, c * 1.0, color[i % 7], 1, 8, false);

}

// 任意の姿勢の体節線の描画
void drawLimb(Mat postures, int sample, int person, Mat image)
{
	// 体節線の数だけ繰り返し
	for (int i = 0; i < 15; i++) {

		// 両端の関節番号の取得
		int p0 = limb[i][0];
		int p1 = limb[i][1];

		// 関節位置の取得
		const auto x0 = postures.at<float>(sample, 3 * p0);
		const auto y0 = postures.at<float>(sample, 3 * p0 + 1);
		const auto c0 = postures.at<float>(sample, 3 * p0 + 2);
		const auto x1 = postures.at<float>(sample, 3 * p1);
		const auto y1 = postures.at<float>(sample, 3 * p1 + 1);
		const auto c1 = postures.at<float>(sample, 3 * p1 + 2);

		// どちらの関節の信頼度も許容最小値以上ならば描画
		if (c0 > MINIMUM_CONFIDENCE && c1 > MINIMUM_CONFIDENCE) {
			cv::line(image, cv::Point(x0, y0), cv::Point(x1, y1), color[person % 7], 2, 8, 0);
		}

	}

}

// クラスタ中心表示用ウィンドウの消去
void removeWindowsForClusteredCenters(int numberOfClusters)
{

	for (int i = 0; i < numberOfClusters; i++) {

		// 表示用ウィンドウ名の設定
		char window_name[20];
		sprintf_s(window_name, 20, "Cluster%d", i);

		// ウィンドウに表示
		destroyWindow(window_name);

	}

}