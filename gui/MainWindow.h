#pragma once

#include "BackendClient.h"

#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPoint>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QSplitter>
#include <QTableWidget>
#include <QTabWidget>
#include <QToolButton>
#include <QComboBox>
#include <QLineEdit>

class ImageViewer : public QScrollArea {
    Q_OBJECT

public:
    explicit ImageViewer(QWidget* parent = nullptr);

    void setImagePath(const QString& imagePath);
    void setZoomFactor(double zoomFactor);
    void setFitToWindow(bool fitToWindow);
    double zoomFactor() const;
    bool fitToWindow() const;
    void zoomIn();
    void zoomOut();
    void resetView();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void refreshPixmap();
    void applyPanDelta(const QPoint& delta);

    QLabel* imageLabel_ = nullptr;
    QString imagePath_;
    QPixmap originalPixmap_;
    double zoomFactor_ = 1.0;
    bool fitToWindow_ = true;
    bool panning_ = false;
    QPoint lastMousePos_;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void openFolder();
    void openFiles();
    void analyzeCurrentSelection();
    void gallerySelectionChanged();
    void applyGalleryFilter(const QString& filterText);
    void backendStarted(const QString& imagePath);
    void backendCompleted();
    void backendFailed(const QString& errorText);
    void stepSelectionChanged();
    void zoomSliderChanged(int value);
    void fitToggleChanged(bool checked);
    void zoomIn();
    void zoomOut();
    void resetZoom();

private:
    void buildUi();
    void setImageList(const QStringList& files);
    void updateGalleryIcons();
    void updateFromPayload(const BackendClient::AnalysisPayload& payload);
    void updateResultsTable(const BackendClient::AnalysisPayload& payload);
    void updateTimingsTable(const BackendClient::AnalysisPayload& payload);
    void updateStepList(const BackendClient::AnalysisPayload& payload);
    void syncZoomControlsFromViewer();
    QString currentImagePath() const;

    BackendClient backend_;
    QListWidget* galleryList_ = nullptr;
    QListWidget* stepList_ = nullptr;
    ImageViewer* imageViewer_ = nullptr;
    QTableWidget* resultsTable_ = nullptr;
    QTableWidget* timingsTable_ = nullptr;
    QPlainTextEdit* logView_ = nullptr;
    QPushButton* analyzeButton_ = nullptr;
    QComboBox* modeCombo_ = nullptr;
    QLineEdit* galleryFilterEdit_ = nullptr;
    QSlider* zoomSlider_ = nullptr;
    QToolButton* fitButton_ = nullptr;
    QLabel* currentImageLabel_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QStringList imagePaths_;
};
