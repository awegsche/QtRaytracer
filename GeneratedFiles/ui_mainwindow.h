/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDial>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout_2;
    QFrame *frame;
    QVBoxLayout *verticalLayout_2;
    QPushButton *pushButton;
    QLabel *label_info;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout;
    QToolBox *toolBox;
    QWidget *page;
    QVBoxLayout *verticalLayout;
    QWidget *Camera;
    QGroupBox *groupBox;
    QDoubleSpinBox *camDirZ;
    QDoubleSpinBox *camPosZ;
    QLabel *label_2;
    QLabel *label;
    QDoubleSpinBox *camDirY;
    QDoubleSpinBox *camPosY;
    QDoubleSpinBox *camDirX;
    QDoubleSpinBox *camPosX;
    QGroupBox *groupBox_2;
    QLabel *angleValue;
    QSlider *angleSlider;
    QLabel *label_3;
    QLabel *label_5;
    QSlider *distanceSlider;
    QLabel *distanceValue;
    QLabel *label_6;
    QDial *dial;
    QLabel *apertureValue;
    QPushButton *pushButton_3;
    QGroupBox *groupBox_3;
    QLabel *label_4;
    QSpinBox *supersamplingBox;
    QGroupBox *groupBox_4;
    QCheckBox *checkBox;
    QLabel *label_9;
    QSlider *hazeSlider;
    QLabel *hazeValue;
    QLabel *label_10;
    QSlider *hazeattenuationSlider;
    QLabel *hazeattenuationValue;
    QWidget *page_2;
    QLabel *label_7;
    QSpinBox *spinBox_width;
    QSpinBox *spinBox_height;
    QLabel *label_8;
    QSpinBox *spinBox_width_2;
    QSpinBox *spinBox_height_2;
    QPushButton *pushButton_2;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1305, 654);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        horizontalLayout_2 = new QHBoxLayout(centralWidget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        frame = new QFrame(centralWidget);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setMaximumSize(QSize(2560, 2560));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frame);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        pushButton = new QPushButton(frame);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setFocusPolicy(Qt::NoFocus);

        verticalLayout_2->addWidget(pushButton);

        label_info = new QLabel(frame);
        label_info->setObjectName(QStringLiteral("label_info"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_info->sizePolicy().hasHeightForWidth());
        label_info->setSizePolicy(sizePolicy1);
        label_info->setMinimumSize(QSize(640, 0));

        verticalLayout_2->addWidget(label_info);


        horizontalLayout_2->addWidget(frame);

        frame_2 = new QFrame(centralWidget);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame_2);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        toolBox = new QToolBox(frame_2);
        toolBox->setObjectName(QStringLiteral("toolBox"));
        page = new QWidget();
        page->setObjectName(QStringLiteral("page"));
        page->setGeometry(QRect(0, 0, 614, 491));
        verticalLayout = new QVBoxLayout(page);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        toolBox->addItem(page, QStringLiteral("Page 1"));
        Camera = new QWidget();
        Camera->setObjectName(QStringLiteral("Camera"));
        Camera->setGeometry(QRect(0, 0, 614, 491));
        groupBox = new QGroupBox(Camera);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 0, 501, 101));
        sizePolicy.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy);
        QFont font;
        font.setBold(false);
        font.setWeight(50);
        groupBox->setFont(font);
        camDirZ = new QDoubleSpinBox(groupBox);
        camDirZ->setObjectName(QStringLiteral("camDirZ"));
        camDirZ->setEnabled(false);
        camDirZ->setGeometry(QRect(320, 60, 81, 22));
        camDirZ->setMinimum(-10000);
        camDirZ->setMaximum(10000);
        camPosZ = new QDoubleSpinBox(groupBox);
        camPosZ->setObjectName(QStringLiteral("camPosZ"));
        camPosZ->setGeometry(QRect(320, 30, 81, 22));
        camPosZ->setMinimum(-10000);
        camPosZ->setMaximum(10000);
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(20, 60, 111, 21));
        QFont font1;
        font1.setBold(true);
        font1.setWeight(75);
        label_2->setFont(font1);
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(20, 30, 61, 20));
        label->setFont(font1);
        camDirY = new QDoubleSpinBox(groupBox);
        camDirY->setObjectName(QStringLiteral("camDirY"));
        camDirY->setEnabled(false);
        camDirY->setGeometry(QRect(230, 60, 81, 22));
        camDirY->setMinimum(-10000);
        camDirY->setMaximum(10000);
        camPosY = new QDoubleSpinBox(groupBox);
        camPosY->setObjectName(QStringLiteral("camPosY"));
        camPosY->setGeometry(QRect(230, 30, 81, 22));
        camPosY->setMinimum(-10000);
        camPosY->setMaximum(10000);
        camDirX = new QDoubleSpinBox(groupBox);
        camDirX->setObjectName(QStringLiteral("camDirX"));
        camDirX->setEnabled(false);
        camDirX->setGeometry(QRect(140, 60, 81, 22));
        camDirX->setMinimum(-10000);
        camDirX->setMaximum(10000);
        camPosX = new QDoubleSpinBox(groupBox);
        camPosX->setObjectName(QStringLiteral("camPosX"));
        camPosX->setGeometry(QRect(140, 30, 81, 22));
        camPosX->setMinimum(-10000);
        camPosX->setMaximum(10000);
        groupBox_2 = new QGroupBox(Camera);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 120, 501, 191));
        angleValue = new QLabel(groupBox_2);
        angleValue->setObjectName(QStringLiteral("angleValue"));
        angleValue->setGeometry(QRect(430, 70, 58, 14));
        angleSlider = new QSlider(groupBox_2);
        angleSlider->setObjectName(QStringLiteral("angleSlider"));
        angleSlider->setGeometry(QRect(110, 70, 311, 20));
        angleSlider->setMinimum(1);
        angleSlider->setMaximum(10000);
        angleSlider->setOrientation(Qt::Horizontal);
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(20, 70, 111, 21));
        label_3->setFont(font1);
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(20, 40, 111, 21));
        label_5->setFont(font1);
        distanceSlider = new QSlider(groupBox_2);
        distanceSlider->setObjectName(QStringLiteral("distanceSlider"));
        distanceSlider->setGeometry(QRect(110, 40, 311, 16));
        distanceSlider->setMinimum(1);
        distanceSlider->setMaximum(10000);
        distanceSlider->setOrientation(Qt::Horizontal);
        distanceValue = new QLabel(groupBox_2);
        distanceValue->setObjectName(QStringLiteral("distanceValue"));
        distanceValue->setGeometry(QRect(430, 40, 58, 14));
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(20, 100, 111, 21));
        label_6->setFont(font1);
        dial = new QDial(groupBox_2);
        dial->setObjectName(QStringLiteral("dial"));
        dial->setGeometry(QRect(140, 90, 50, 64));
        dial->setMaximum(1000);
        dial->setSingleStep(1);
        apertureValue = new QLabel(groupBox_2);
        apertureValue->setObjectName(QStringLiteral("apertureValue"));
        apertureValue->setGeometry(QRect(150, 150, 31, 16));
        pushButton_3 = new QPushButton(groupBox_2);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));
        pushButton_3->setGeometry(QRect(280, 130, 75, 23));
        groupBox_3 = new QGroupBox(Camera);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(530, 10, 191, 301));
        label_4 = new QLabel(groupBox_3);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(10, 30, 111, 20));
        label_4->setFont(font1);
        supersamplingBox = new QSpinBox(groupBox_3);
        supersamplingBox->setObjectName(QStringLiteral("supersamplingBox"));
        supersamplingBox->setGeometry(QRect(120, 30, 61, 22));
        supersamplingBox->setMinimum(1);
        supersamplingBox->setMaximum(10000);
        supersamplingBox->setValue(4);
        groupBox_4 = new QGroupBox(Camera);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        groupBox_4->setGeometry(QRect(20, 320, 691, 141));
        checkBox = new QCheckBox(groupBox_4);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        checkBox->setGeometry(QRect(20, 30, 76, 21));
        checkBox->setFont(font1);
        label_9 = new QLabel(groupBox_4);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(90, 30, 71, 21));
        label_9->setFont(font1);
        hazeSlider = new QSlider(groupBox_4);
        hazeSlider->setObjectName(QStringLiteral("hazeSlider"));
        hazeSlider->setGeometry(QRect(170, 30, 311, 16));
        hazeSlider->setMinimum(1);
        hazeSlider->setMaximum(50000);
        hazeSlider->setOrientation(Qt::Horizontal);
        hazeValue = new QLabel(groupBox_4);
        hazeValue->setObjectName(QStringLiteral("hazeValue"));
        hazeValue->setGeometry(QRect(490, 30, 58, 14));
        label_10 = new QLabel(groupBox_4);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(90, 60, 81, 21));
        label_10->setFont(font1);
        hazeattenuationSlider = new QSlider(groupBox_4);
        hazeattenuationSlider->setObjectName(QStringLiteral("hazeattenuationSlider"));
        hazeattenuationSlider->setGeometry(QRect(170, 60, 311, 16));
        hazeattenuationSlider->setMinimum(1);
        hazeattenuationSlider->setMaximum(50000);
        hazeattenuationSlider->setOrientation(Qt::Horizontal);
        hazeattenuationValue = new QLabel(groupBox_4);
        hazeattenuationValue->setObjectName(QStringLiteral("hazeattenuationValue"));
        hazeattenuationValue->setGeometry(QRect(490, 60, 58, 14));
        toolBox->addItem(Camera, QStringLiteral("Camera"));
        page_2 = new QWidget();
        page_2->setObjectName(QStringLiteral("page_2"));
        page_2->setGeometry(QRect(0, 0, 614, 491));
        label_7 = new QLabel(page_2);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(20, 20, 61, 20));
        label_7->setFont(font1);
        spinBox_width = new QSpinBox(page_2);
        spinBox_width->setObjectName(QStringLiteral("spinBox_width"));
        spinBox_width->setGeometry(QRect(60, 20, 61, 22));
        spinBox_width->setMinimum(32);
        spinBox_width->setMaximum(2560);
        spinBox_width->setSingleStep(16);
        spinBox_width->setValue(640);
        spinBox_height = new QSpinBox(page_2);
        spinBox_height->setObjectName(QStringLiteral("spinBox_height"));
        spinBox_height->setGeometry(QRect(130, 20, 61, 22));
        spinBox_height->setMinimum(32);
        spinBox_height->setMaximum(2560);
        spinBox_height->setSingleStep(16);
        spinBox_height->setValue(480);
        label_8 = new QLabel(page_2);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(20, 80, 61, 20));
        label_8->setFont(font1);
        spinBox_width_2 = new QSpinBox(page_2);
        spinBox_width_2->setObjectName(QStringLiteral("spinBox_width_2"));
        spinBox_width_2->setGeometry(QRect(100, 80, 61, 22));
        spinBox_width_2->setMinimum(32);
        spinBox_width_2->setMaximum(2560);
        spinBox_width_2->setSingleStep(16);
        spinBox_width_2->setValue(640);
        spinBox_height_2 = new QSpinBox(page_2);
        spinBox_height_2->setObjectName(QStringLiteral("spinBox_height_2"));
        spinBox_height_2->setGeometry(QRect(170, 80, 61, 22));
        spinBox_height_2->setMinimum(32);
        spinBox_height_2->setMaximum(2560);
        spinBox_height_2->setSingleStep(16);
        spinBox_height_2->setValue(480);
        pushButton_2 = new QPushButton(page_2);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));
        pushButton_2->setGeometry(QRect(260, 80, 121, 23));
        toolBox->addItem(page_2, QStringLiteral("Image"));

        horizontalLayout->addWidget(toolBox);


        horizontalLayout_2->addWidget(frame_2);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1305, 19));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);
        QObject::connect(angleSlider, SIGNAL(sliderMoved(int)), angleValue, SLOT(setNum(int)));

        toolBox->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        pushButton->setText(QApplication::translate("MainWindow", "save image", Q_NULLPTR));
        label_info->setText(QApplication::translate("MainWindow", "TextLabel", Q_NULLPTR));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("MainWindow", "Page 1", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("MainWindow", "Position", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", "View direction:", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "Position:", Q_NULLPTR));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Lens", Q_NULLPTR));
        angleValue->setText(QApplication::translate("MainWindow", "dist value", Q_NULLPTR));
        label_3->setText(QApplication::translate("MainWindow", "Angle", Q_NULLPTR));
        label_5->setText(QApplication::translate("MainWindow", "Focus", Q_NULLPTR));
        distanceValue->setText(QApplication::translate("MainWindow", "focus_value", Q_NULLPTR));
        label_6->setText(QApplication::translate("MainWindow", "Aperture", Q_NULLPTR));
        apertureValue->setText(QApplication::translate("MainWindow", "value", Q_NULLPTR));
        pushButton_3->setText(QApplication::translate("MainWindow", "PushButton", Q_NULLPTR));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "GroupBox", Q_NULLPTR));
        label_4->setText(QApplication::translate("MainWindow", "Supersampling:", Q_NULLPTR));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "World", Q_NULLPTR));
        checkBox->setText(QApplication::translate("MainWindow", "Haze", Q_NULLPTR));
        label_9->setText(QApplication::translate("MainWindow", "Distance:", Q_NULLPTR));
        hazeValue->setText(QApplication::translate("MainWindow", "focus_value", Q_NULLPTR));
        label_10->setText(QApplication::translate("MainWindow", "Attenuation:", Q_NULLPTR));
        hazeattenuationValue->setText(QApplication::translate("MainWindow", "focus_value", Q_NULLPTR));
        toolBox->setItemText(toolBox->indexOf(Camera), QApplication::translate("MainWindow", "Camera", Q_NULLPTR));
        label_7->setText(QApplication::translate("MainWindow", "Size:", Q_NULLPTR));
        label_8->setText(QApplication::translate("MainWindow", "Big size:", Q_NULLPTR));
        pushButton_2->setText(QApplication::translate("MainWindow", "render big image", Q_NULLPTR));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("MainWindow", "Image", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
