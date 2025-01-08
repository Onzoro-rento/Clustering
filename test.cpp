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

#define IMAGE_WIDTH 640        // RGB-D�J�����̃t���[���摜�̕�
#define IMAGE_HEIGHT 360       // RGB-D�J�����̃t���[���摜�̍���
#define NUMBER_OF_JOINTS 25    // �֐ߐ�
#define MINIMUM_CONFIDENCE 0.1 // �M���x�̋��e�ŏ��l
#define KNN_NUMBER 3           // kNN�@�̍ŋߖT��

// �֐��v���g�^�C�v�錾
int clusteringPostures(Mat posturesForClustering, int numberOfClusters, Mat*clusteredLabels);
void drawJoint(int i, int j, float c, float x, float y, Mat plane);
void drawLimb(Mat postures, int sample, int person, Mat image);
void removeWindowsForClusteredCenters(int number_of_clusters);

// �l���\���F�̐ݒ�
Scalar color[7] =
{ Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),
  Scalar(0,255,255),Scalar(255,0,255),Scalar(255,255,0),Scalar(255,255,255) };

// �e�̐߂̊֐ߑ�
int limb[15][2] =
{ {1,2},{2,3},{3,4},{1,5},{5,6},{6,7},{1,8},{8,9},
  {8,12},{9,10},{10,11},{12,13},{13,14},{11,22},{14,20} };
#include <map> // �N���X�^ID�ɑΉ�����p���`�����Ǘ�

// �p���`���Ƃ��̃J�E���g
std::map<int, std::string> clusterToPunchType;
std::map<std::string, int> punchCounts;


// �N���X�^���̏������֐�



int main(int argc, char *argv[])
{
	// --------------�r�f�I�ǂݍ���----------------------------------
	int flag = 0;
	cv::VideoCapture video("sample.mp4");
	cv::VideoCapture video2("test.mp4");
	if (!video.isOpened()) { // �G���[����
		std::cout << "video.error" << std::endl;
		return -1;
	}
	if (!video2.isOpened()) { // �G���[����
		std::cout << "video.error" << std::endl;
		return -1;
	}
	Mat colorImage;

	cv::Mat frame, gray, canny; // �t���[���i�[�p

	int width = video.get(cv::CAP_PROP_FRAME_WIDTH); // ���悩�畝���擾
	int height = video.get(cv::CAP_PROP_FRAME_HEIGHT); // ���悩�獂�����擾
	int count = 0; //�p���`�̐��𐔂���
	int presponse = -1; //�O���response�̒l
	int presponse2 = -2;//�O�X���responce�̒l
	float prevLeftHandX = 0, prevLeftHandY = 0; // ����̑O��ʒu
	float prevRightHandX = 0, prevRightHandY = 0; // �E��̑O��ʒu
	const float punchThreshold = 30.0; // �p���`���o�̈ړ��ʂ������l

	// <<<---------- RGB - D�J�������g���ꍇ�̏����������������� ----------

	// �����摜���J���[�摜�ɍ��킹�邽�߂̃p�����[�^�ݒ�
	//rs2::align alignToColor(RS2_STREAM_COLOR);

	// �J���[�摜�Ƌ����摜�̉f���X�g���[���̊K�����ƃt���[�����[�g�ݒ�
	//config streamConfig;
	/*streamConfig.enable_stream(RS2_STREAM_COLOR, IMAGE_WIDTH, IMAGE_HEIGHT, RS2_FORMAT_BGR8, 30);
	streamConfig.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_Z16, 30);
*/
// �f���X�g���[�~���O�J�n
///pipeline pipeLine;
//auto profile = pipeLine.start(streamConfig);

// ---------- RGB - D�J�������g���ꍇ�̏��������������܂� ---------->>>

// <<<---------- �ʏ�J�������g���ꍇ�̏����������������� ----------

//VideoCapture camera;
//camera.open(0); // �J�����ԍ��͓K���Ȃ��̂��Z�b�g

// ---------- �ʏ�J�������g���ꍇ�̏��������������܂� ---------->>>

// <<<---------- OpenPose���g�����߂̏����������������� ----------

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

	// ---------- OpenPose���g�����߂̏��������������܂� ---------->>>

	// <<<---------- �N���X�^�����O�̂��߂̏����������������� ----------

	// �N���X�^�����O�Ώێp���̕ۑ��p�z��̐錾�i�ŏ��͎b��I�ȂP�T���v���̂ݕۑ��\�ȃT�C�Y <-- Mat�^�̃T�C�Y�L�q���@�ɒ��ӁFMat (int rows, int cols, int type) �j
	Mat posturesForClustering = Mat(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);
	Mat posturesForClustering2 = Mat(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);

	int storedSamples = 0;	// �i�[�σf�[�^���i�ŏ���0�j
	int save = -1;			// �f�[�^�ۑ��̗L���i�ŏ��͕ۑ��Ȃ����[�h�j

	// �N���X�^���w��p�g���b�N�o�[�̐ݒu
	namedWindow("Color Image");
	int number_of_clusters = 0; // �ύX�\�ȃN���X�^��
	int clusters = 3;
	createTrackbar("Clusters", "Color Image", &clusters, 10);

	// ---------- �N���X�^�����O�̂��߂̏��������������܂� ---------->>>

	// <<<---------- kNN�@�̂��߂̏����������������� ----------

	Ptr<ml::KNearest> knn = ml::KNearest::create();				// kNN���ʊ�̐���
	knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);	// �ߖT�T���@�̐ݒ�
	knn->setDefaultK(KNN_NUMBER);								// �ߖT���̐ݒ�
	knn->setIsClassifier(true);									// ���ʌ��ʂ̓J�e�S��
	int recognize = -1;	
	auto lastIncrementTime = std::chrono::steady_clock::now();
	const int delayInMillis = 150; // �C���N�������g�Ԃ̑ҋ@���ԁi�~���b�j// kNN�ɂ��F���̗L���i�ŏ��͔F���Ȃ����[�h�j

	// ---------- kNN�@�̂��߂̏��������������܂� ---------->>>

	while (1) {
		

		if (flag == 0)
		{
			video >> colorImage;
		}

		if (flag == 0 && colorImage.empty()) {
			flag = 1;
			if (number_of_clusters > 0) removeWindowsForClusteredCenters(number_of_clusters);

			number_of_clusters = clusters;

			Mat clusteredLabels;  // �N���X�^�����O�p�̊e�T���v���̃N���X�i�[�p�z��̐錾

			int result_of_clustering = clusteringPostures(posturesForClustering, number_of_clusters, &clusteredLabels);  // �N���X�^�����O����(�N���X�^�����O���ʎ󂯎��)
			save = -1;  // �T���v���ۑ��͒��~

			if (result_of_clustering == 0) {
				knn->train(posturesForClustering, ml::ROW_SAMPLE, clusteredLabels);  // kNN�@�̃T���v���̊w�K
				recognize = 1; // �F�����[�hON
				
			}
			else recognize = -1;
		}
		if (flag == 1) { video2 >> colorImage; }
		if (flag == 1 && colorImage.empty()) { break; }
		// <<<---------- RGB-D�J�������g���ꍇ�̃t���[���摜�擾������������ ----------

		// RGB-D�J�������g���ꍇ�̃J���[�摜�Ƌ����摜�̃t���[���Z�b�g�̎擾
		//frameset frameSet = video.wait_for_frames();

		//// �����摜���J���[�摜�ɍ��킹��
		//frameSet = alignToColor.process(frameSet);

		//// �t���[���Z�b�g����J���[�摜�Ƌ����摜�̃t���[�����l��
		//video_frame colorFrame = frameSet.get_color_frame();
		//depth_frame depthFrame = frameSet.get_depth_frame();

		//// Mat�^�̃J���[�摜�Ƌ����摜�f�[�^���l��
		//Mat colorImage(Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3, (void*)colorFrame.get_data(), Mat::AUTO_STEP);
		//Mat depthImage16(Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_16UC1, (void*)depthFrame.get_data(), Mat::AUTO_STEP);

		// ---------- RGB-D�J�������g���ꍇ�̃t���[���摜�擾���������܂� ---------->>>

		// <<<---------- �ʏ�̃J���[�J�������g���ꍇ�̃t���[���摜�̎擾�������� ----------

		//Mat colorImage;
		//camera >> colorImage;
		//resize(colorImage, colorImage, Size(IMAGE_WIDTH, IMAGE_HEIGHT));

		// ---------- �ʏ�̃J���[�J�������g���ꍇ�̃t���[���摜�̎擾�����܂� ---------->>>


		// �p���f�[�^�E�N���X�^���x���i�[�p�z��ւ̍s�ǉ�
		if (posturesForClustering.rows <= storedSamples)posturesForClustering.resize(storedSamples + 1);

		// <<<---------- OpenPose�ɂ��p�����菈���������� ----------

		const op::Point<int> imageSize{ colorImage.cols, colorImage.rows };
		vector<double> scaleInputToNetInputs;
		vector<op::Point<int>> netInputSizes;
		double scaleInputToOutput;
		op::Point<int> outputResolution;
		tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
			= scaleAndSizeExtractor.extract(imageSize);
		const auto netInputArray = cvMatToOpInput.createArray(colorImage, scaleInputToNetInputs, netInputSizes);
		poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);

		// ---------- OpenPose�ɂ��p�����菈�������܂� ---------->>>


		// <<<---------- ���茋�ʂ̎p���\�������������� ----------

		const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
		const auto numberPeopleDetected = poseKeypoints.getSize(0);
		const auto numberBodyParts = poseKeypoints.getSize(1);

		// �l�����̏���
		for (int i = 0; i < numberPeopleDetected; i++) {

			// �Ώېl���̐���p���f�[�^�i�[�p�z��̐錾
			Mat observedPosture(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);
			Mat observedPosture2(1, 3 * NUMBER_OF_JOINTS, CV_32FC1);
			float leftHandX = poseKeypoints[{i, 7, 0}] - poseKeypoints[{i, 1, 0}]; // �����
			float leftHandY = poseKeypoints[{i, 7, 1}] - poseKeypoints[{i, 1, 1}];
			float rightHandX = poseKeypoints[{i, 4, 0}] - poseKeypoints[{i, 1, 0}]; // �E���
			float rightHandY = poseKeypoints[{i, 4, 1}] - poseKeypoints[{i, 1, 1}];
			
			float leftMovement = sqrt(pow(leftHandX - prevLeftHandX, 2) + pow(leftHandY - prevLeftHandY, 2));
			float rightMovement = sqrt(pow(rightHandX - prevRightHandX, 2) + pow(rightHandY - prevRightHandY, 2));
			auto currentTime = std::chrono::steady_clock::now();
			// �O��̃C���N�������g����̌o�ߎ��Ԃ��v�Z
			auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastIncrementTime).count();

			// �֐ߖ��̏���
			for (int j = 0; j < numberBodyParts; j++) {

				// i�Ԗڂ̐l����j�Ԗڂ̊֐߈ʒu(x,y)�ƐM���xc�̎擾

				const auto x = poseKeypoints[{i, j, 0}] - poseKeypoints[{i, 8, 0}];
				const auto y = poseKeypoints[{i, j, 1}] - poseKeypoints[{i, 8, 1}];
				const auto c = poseKeypoints[{i, j, 2}];



				// i�Ԗڂ̐l����j�Ԗڂ̊֐߈ʒu(x,y)���C�M���xc�ɔ�Ⴕ�����a�̉~��colorImage��ɕ`��
				drawJoint(i, j, c, x + poseKeypoints[{i, 8, 0}], y + poseKeypoints[{i, 8, 1}], colorImage);

				// ��̊֐߈ʒu�ƐM���x�𐄒�p���f�[�^�i�[�p�z��Ɋi�[


				observedPosture.at<float>(3 * j) = x;
				observedPosture.at<float>(3 * j + 1) = y;
				observedPosture.at<float>(3 * j + 2) = c;
				observedPosture2.at<float>(3 * j) = x + poseKeypoints[{i, 8, 0}];
				observedPosture2.at<float>(3 * j + 1) = y + poseKeypoints[{i, 8, 1}];
				observedPosture2.at<float>(3 * j + 2) = c;


				// 0�Ԗڂ̐l���̏ꍇ
				if (i == 0 && flag == 0) {
					// ��̊֐߈ʒu�ƐM���x���N���X�^�����O�Ώێp���̕ۑ��p�z��ɂ��ۑ�
					posturesForClustering.at<float>(storedSamples, 3 * j) = x;
					posturesForClustering.at<float>(storedSamples, 3 * j + 1) = y;
					posturesForClustering.at<float>(storedSamples, 3 * j + 2) = c;

				}

			}

			// ����p���f�[�^�i�[�p�z����g���đ̐ߐ��̕`��
			drawLimb(observedPosture2, 0, i, colorImage);

			// �ۑ����[�h��ON��0�Ԗڂ̐l���̏ꍇ�̓N���X�^�����O�Ώێp���̕ۑ��p�z��ɕۑ����ꂽ�p���T���v�����̃J�E���g�A�b�v
			if (save > 0 && i == 0 && flag == 0) {
				storedSamples++;
				printf("%d samples have stored\n", storedSamples);
			}

			// �F�����[�hON ���� 0�Ԗڂ̐l���̏ꍇ
			if (i == 0 && recognize > 0 && flag == 1) {

				// ����p���f�[�^�i�[�p�z��𗘗p���ĔF���E�\��

				int response = (int)knn->predict(observedPosture);

				
				
				
				

				if ((response != presponse) && (presponse != presponse2) && (leftMovement > punchThreshold || rightMovement > punchThreshold) &&
					(elapsedTime > delayInMillis)) {
					

					
						count++;
						lastIncrementTime = currentTime; // �C���N�������g�������_�̎������L�^
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

		// ---------- ���茋�ʂ̎p���\�����������܂� ---------->>>

		imshow("Color Image", colorImage);


		int key = cv::waitKey(1);

		// q�L�[�ŏI��
		if (key == 'q') break;

		// s�L�[�ŃN���X�^�����O�p�̃T���v���ۑ�ON/OFF
		else if (key == 's') save *= -1;

		// c�L�[�ŃN���X�^�����O


		// r�L�[�Ń��Z�b�g
		else if (key == 'r') {

			// �N���X�^�����O�p�̕ۑ��T���v��������
			storedSamples = 0;
			posturesForClustering.resize(1);

			// �N���X�^�����O���ʂ̏���
			if (number_of_clusters > 0) removeWindowsForClusteredCenters(number_of_clusters);

			save = -1;  // �T���v���ۑ��͒��~
			recognize = -1; // �F�����[�hOFF

		}

	}

	return 0;

}

// �N���X�^�����O
int clusteringPostures(Mat posturesForClustering, int numberOfClusters, Mat *labels)
{
	// �N���X�^����0�̂Ƃ��͉������Ȃ�
	if (numberOfClusters == 0) return -1;

	// <<<---------- �N���X�^�����O������������ ----------

	// ���ރN���X�^�C�N���X�^���S�̏o�͗p�z��̐錾
	Mat centers;

	// k���ϖ@�̎��s
	kmeans(posturesForClustering, numberOfClusters, *labels,
		TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 1, KMEANS_PP_CENTERS, centers);

	for (int k = 0; k < (*labels).rows; k++) {
		int cluster = (*labels).at<int>(k);
		printf("Sample_%d = Cluster_%d\n", k, cluster);
	}

	// ---------- �N���X�^�����O���������܂� ---------->>>

	// <<<---------- �N���X�^���S�̕`�悱������ ----------

	// �N���X�^�������J��Ԃ�
	for (int i = 0; i < numberOfClusters; i++) {

		// �`��p�摜�����ɓh��Ԃ�
		Mat centroidImage = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);

		// �֐ߐ������J��Ԃ�
		for (int j = 0; j < NUMBER_OF_JOINTS; j++) {

			// �֐߈ʒu�̊l��
			float x, y, c;
			x = centers.at<float>(i, 3 * j);
			y = centers.at<float>(i, 3 * j + 1);
			c = centers.at<float>(i, 3 * j + 2);
			drawJoint(i, j, c, x + 300, y + 300, centroidImage);
		}

		drawLimb(centers + 300, i, i, centroidImage);

		// �\���p�E�B���h�E���̐ݒ�
		char window_name[20];
		sprintf_s(window_name, 20, "Cluster%d", i);

		// �E�B���h�E�ɕ\��
		imshow(window_name, centroidImage);

	}

	// ---------- �N���X�^���S�̕`�悱���܂� ---------->>>

	return 0;

}

// �C�ӂ̊֐߈ʒu�̕`��
void drawJoint(int i, int j, float c, float x, float y, Mat plane)
{

	cv::Point p = cv::Point(x, y);
	circle(plane, p, 5 * c, color[i % 7], -1, 8, 0);
	putText(plane, to_string(j), p, FONT_HERSHEY_COMPLEX_SMALL, c * 1.0, color[i % 7], 1, 8, false);

}

// �C�ӂ̎p���̑̐ߐ��̕`��
void drawLimb(Mat postures, int sample, int person, Mat image)
{
	// �̐ߐ��̐������J��Ԃ�
	for (int i = 0; i < 15; i++) {

		// ���[�̊֐ߔԍ��̎擾
		int p0 = limb[i][0];
		int p1 = limb[i][1];

		// �֐߈ʒu�̎擾
		const auto x0 = postures.at<float>(sample, 3 * p0);
		const auto y0 = postures.at<float>(sample, 3 * p0 + 1);
		const auto c0 = postures.at<float>(sample, 3 * p0 + 2);
		const auto x1 = postures.at<float>(sample, 3 * p1);
		const auto y1 = postures.at<float>(sample, 3 * p1 + 1);
		const auto c1 = postures.at<float>(sample, 3 * p1 + 2);

		// �ǂ���̊֐߂̐M���x�����e�ŏ��l�ȏ�Ȃ�Ε`��
		if (c0 > MINIMUM_CONFIDENCE && c1 > MINIMUM_CONFIDENCE) {
			cv::line(image, cv::Point(x0, y0), cv::Point(x1, y1), color[person % 7], 2, 8, 0);
		}

	}

}

// �N���X�^���S�\���p�E�B���h�E�̏���
void removeWindowsForClusteredCenters(int numberOfClusters)
{

	for (int i = 0; i < numberOfClusters; i++) {

		// �\���p�E�B���h�E���̐ݒ�
		char window_name[20];
		sprintf_s(window_name, 20, "Cluster%d", i);

		// �E�B���h�E�ɕ\��
		destroyWindow(window_name);

	}

}