#pragma once

#include "BackendClient.h"

#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QMap>
#include <QColor>
#include <QCheckBox>
#include <QDoubleSpinBox>
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
    struct AnnotationBox {
        QString label;
        QRect imageRect;
        QColor color;
    };

    struct SectorAnnotation {
        QString label;
        QColor color;
        double startAngleDeg = 0.0;
        double endAngleDeg = 0.0;
    };

    enum class InteractionMode {
        View,
        DrawSizeSector,
        DrawDotSector
    };

    explicit ImageViewer(QWidget* parent = nullptr);

    void setImagePath(const QString& imagePath);
    void setZoomFactor(double zoomFactor);
    void setFitToWindow(bool fitToWindow);
    double zoomFactor() const;
    bool fitToWindow() const;
    void zoomIn();
    void zoomOut();
    void resetView();
    void setAnnotationBoxes(const QVector<AnnotationBox>& boxes);
    void setSectorAnnotations(const QVector<SectorAnnotation>& sectors);
    void setInteractionMode(InteractionMode mode);
    void setWheelGeometry(bool found, const QPointF& center, double innerRadius, double outerRadius);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void refreshPixmap();
    void applyPanDelta(const QPoint& delta);
    QRect displayedImageRect() const;
    QPoint mapLabelPointToImage(const QPoint& labelPoint) const;
    double angleForImagePoint(const QPoint& imagePoint) const;

    QLabel* imageLabel_ = nullptr;
    QString imagePath_;
    QPixmap originalPixmap_;
    double zoomFactor_ = 1.0;
    bool fitToWindow_ = true;
    bool panning_ = false;
    QPoint lastMousePos_;
    QVector<AnnotationBox> annotationBoxes_;
    QVector<SectorAnnotation> sectorAnnotations_;
    InteractionMode interactionMode_ = InteractionMode::View;
    bool drawingAnnotation_ = false;
    QPoint annotationStartImage_;
    QPoint annotationEndImage_;
    bool wheelGeometryFound_ = false;
    QPointF wheelCenter_;
    double wheelInnerRadius_ = 0.0;
    double wheelOuterRadius_ = 0.0;

signals:
    void annotationDrawn(const QString& label, const QRect& imageRect);
    void sectorDrawn(const QString& label, double startAngleDeg, double endAngleDeg);
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
    void startSizeAnnotation();
    void startDotAnnotation();
    void clearSizeAnnotation();
    void clearDotAnnotation();
    void exportAnnotations();
    void handleAnnotationDrawn(const QString& label, const QRect& imageRect);
    void handleSectorDrawn(const QString& label, double startAngleDeg, double endAngleDeg);
    void sizeCropTopChanged(int value);
    void sizeCropBottomChanged(int value);
    void dotCropTopChanged(int value);
    void dotCropBottomChanged(int value);
    void sizeSectorStartChanged(double value);
    void sizeSectorEndChanged(double value);
    void dotSectorStartChanged(double value);
    void dotSectorEndChanged(double value);
    void applySizeSectorEdits();
    void applyDotSectorEdits();
    void applyWheelOverride();
    void openAnnotationsFile();
    void saveAnnotationsFile();
    void applySizeSuggestion();
    void applyDotSuggestion();

private:
    struct ImageAnnotations {
        double sizeStartAngleDeg = 0.0;
        double sizeEndAngleDeg = 0.0;
        bool sizeSectorSet = false;
        int sizeCropTopPercent = 0;
        int sizeCropBottomPercent = 100;
        double dotStartAngleDeg = 0.0;
        double dotEndAngleDeg = 0.0;
        bool dotSectorSet = false;
        int dotCropTopPercent = 0;
        int dotCropBottomPercent = 100;
        QString sizePreviewPath;
        QString dotPreviewPath;
        QString sizeText;
        QString dotText;
        QString dot4Text;
        bool wheelOverrideSet = false;
        double wheelCenterX = 0.0;
        double wheelCenterY = 0.0;
        double wheelInnerRadius = 0.0;
        double wheelOuterRadius = 0.0;
        bool sizeVisible = true;
        bool dotVisible = true;
        bool dotDateVisible = false;
    };

    struct SuggestionEntry {
        QString sizeBaseGuess;
        QString sizeFullGuess;
        QString sizeConfidence;
        QString sizeLocationHint;
        QString dotTextGuess;
        QString dot4Guess;
        QString dotConfidence;
        QString dotLocationHint;
        QString reviewStatus;
        QString notes;
    };

    void buildUi();
    void setImageList(const QStringList& files);
    void updateGalleryIcons();
    void updateFromPayload(const BackendClient::AnalysisPayload& payload);
    void updateResultsTable(const BackendClient::AnalysisPayload& payload);
    void updateTimingsTable(const BackendClient::AnalysisPayload& payload);
    void updateStepList(const BackendClient::AnalysisPayload& payload);
    void syncZoomControlsFromViewer();
    void refreshAnnotationViewer();
    void refreshPreviewCropBoxes();
    QRect previewCropRect(const QString& previewPath, int topPercent, int bottomPercent) const;
    void syncControlsFromAnnotations(const ImageAnnotations& ann, const BackendClient::AnalysisPayload& payload);
    QString currentImagePath() const;
    QString currentAnnotationImagePath() const;
    QString resolveAnnotationImagePath(const QString& sourceImagePath) const;
    QString resolveWheelAnnotationImagePath(const BackendClient::AnalysisPayload& payload) const;
    QString resolveDefaultUnwrapImagePath(const BackendClient::AnalysisPayload& payload) const;
    QString normalizePathKey(const QString& path) const;
    void loadAnnotationsFromFile(const QString& filePath);
    void saveAnnotationsToFile(const QString& filePath) const;
    void autoSaveAnnotations() const;
    QString defaultAnnotationsPathForCurrentSelection() const;
    QString defaultSuggestionsPathForCurrentSelection() const;
    void loadSuggestionsFromCsv(const QString& filePath);
    void refreshSuggestionPanel();
    QString extractImageNumberKey(const QString& path) const;

    BackendClient backend_;
    QListWidget* galleryList_ = nullptr;
    QListWidget* stepList_ = nullptr;
    ImageViewer* imageViewer_ = nullptr;
    ImageViewer* annotationViewer_ = nullptr;
    ImageViewer* sizePreviewViewer_ = nullptr;
    ImageViewer* dotPreviewViewer_ = nullptr;
    QTableWidget* resultsTable_ = nullptr;
    QTableWidget* timingsTable_ = nullptr;
    QPlainTextEdit* logView_ = nullptr;
    QPlainTextEdit* roiOcrView_ = nullptr;
    QPushButton* analyzeButton_ = nullptr;
    QPushButton* ocrSizeButton_ = nullptr;
    QPushButton* ocrDotButton_ = nullptr;
    QComboBox* modeCombo_ = nullptr;
    QLineEdit* galleryFilterEdit_ = nullptr;
    QLineEdit* sizeTextEdit_ = nullptr;
    QLineEdit* dotTextEdit_ = nullptr;
    QLineEdit* dot4TextEdit_ = nullptr;
    QDoubleSpinBox* sizeSectorStartSpin_ = nullptr;
    QDoubleSpinBox* sizeSectorEndSpin_ = nullptr;
    QDoubleSpinBox* dotSectorStartSpin_ = nullptr;
    QDoubleSpinBox* dotSectorEndSpin_ = nullptr;
    QDoubleSpinBox* wheelCenterXSpin_ = nullptr;
    QDoubleSpinBox* wheelCenterYSpin_ = nullptr;
    QDoubleSpinBox* wheelInnerRadiusSpin_ = nullptr;
    QDoubleSpinBox* wheelOuterRadiusSpin_ = nullptr;
    QSlider* sizeCropTopSlider_ = nullptr;
    QSlider* sizeCropBottomSlider_ = nullptr;
    QSlider* dotCropTopSlider_ = nullptr;
    QSlider* dotCropBottomSlider_ = nullptr;
    QLabel* sizeCropLabel_ = nullptr;
    QLabel* dotCropLabel_ = nullptr;
    QCheckBox* sizeVisibleCheck_ = nullptr;
    QCheckBox* dotVisibleCheck_ = nullptr;
    QCheckBox* dotDateVisibleCheck_ = nullptr;
    QLineEdit* suggestedSizeEdit_ = nullptr;
    QLineEdit* suggestedDotEdit_ = nullptr;
    QLineEdit* suggestedDot4Edit_ = nullptr;
    QLineEdit* suggestedSizeMetaEdit_ = nullptr;
    QLineEdit* suggestedDotMetaEdit_ = nullptr;
    QPlainTextEdit* suggestionNotesView_ = nullptr;
    QPushButton* applySizeSuggestionButton_ = nullptr;
    QPushButton* applyDotSuggestionButton_ = nullptr;
    QSlider* zoomSlider_ = nullptr;
    QToolButton* fitButton_ = nullptr;
    QLabel* currentImageLabel_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QStringList imagePaths_;
    QString annotationsFilePath_;
    QString suggestionsFilePath_;
    QMap<QString, ImageAnnotations> annotationsByImage_;
    QMap<QString, SuggestionEntry> suggestionsByImage_;
    QMap<QString, BackendClient::AnalysisPayload> payloadsBySourceImage_;
};
