#pragma once

#include <QObject>
#include <QProcess>
#include <QRect>
#include <QString>
#include <QStringList>
#include <QVector>

class BackendClient : public QObject {
    Q_OBJECT

public:
    enum class RunMode {
        FullAnalysis,
        WheelGeometry
    };

    struct StepImage {
        QString label;
        QString path;
    };

    struct ResultField {
        QString raw;
        QString normalized;
        bool found = false;
        double confidence = 0.0;
        QString cropPath;
    };

    struct TimingEntry {
        QString name;
        double ms = 0.0;
    };

    struct AnalysisPayload {
        RunMode mode = RunMode::FullAnalysis;
        QString inputPath;
        QString frameId;
        bool wheelFound = false;
        double wheelCenterX = 0.0;
        double wheelCenterY = 0.0;
        double wheelInnerRadius = 0.0;
        double wheelOuterRadius = 0.0;
        QString originalCopyPath;
        QString wheelOverlayPath;
        QString unwrappedBandPath;
        QString overlayPath;
        QString debugDir;
        QString timingReportPath;
        QString ocrReportPath;
        QStringList notes;
        QString stderrText;
        ResultField tyreSize;
        ResultField dot;
        bool tyreSizeFound = false;
        bool dotFound = false;
        bool dotWeekYearFound = false;
        bool dotFullFound = false;
        QString dotWeekYear;
        QString dotFullRaw;
        QString dotFullNormalized;
        QVector<TimingEntry> timings;
        QVector<StepImage> images;
    };

    struct RoiOcrPayload {
        QString imagePath;
        QString branch;
        QRect roi;
        QString rawText;
        QString normalizedText;
        bool found = false;
        double confidence = 0.0;
        QString cropPath;
        QStringList notes;
        QVector<TimingEntry> timings;
    };

    struct SectorPreviewPayload {
        QString imagePath;
        QString branch;
        double startAngleDeg = 0.0;
        double endAngleDeg = 0.0;
        bool wheelFound = false;
        double wheelCenterX = 0.0;
        double wheelCenterY = 0.0;
        double wheelInnerRadius = 0.0;
        double wheelOuterRadius = 0.0;
        QString overlayPath;
        QString unwrappedPath;
        QStringList notes;
        QVector<TimingEntry> timings;
    };

    explicit BackendClient(QObject* parent = nullptr);

    bool isBusy() const;
    bool isRoiBusy() const;
    bool isSectorBusy() const;
    void analyzeImage(const QString& imagePath, RunMode mode);
    void runRoiOcr(const QString& imagePath, const QRect& roi, const QString& branch);
    void runSectorPreview(const QString& imagePath,
                          const QString& branch,
                          double startAngleDeg,
                          double endAngleDeg,
                          bool useWheelOverride = false,
                          double wheelCenterX = 0.0,
                          double wheelCenterY = 0.0,
                          double wheelInnerRadius = 0.0,
                          double wheelOuterRadius = 0.0);
    const AnalysisPayload& lastPayload() const;
    const RoiOcrPayload& lastRoiPayload() const;
    const SectorPreviewPayload& lastSectorPayload() const;
    QString lastError() const;

signals:
    void analysisStarted(const QString& imagePath);
    void analysisCompleted();
    void analysisFailed(const QString& errorText);
    void roiOcrStarted(const QString& imagePath, const QString& branch);
    void roiOcrCompleted();
    void roiOcrFailed(const QString& errorText);
    void sectorPreviewStarted(const QString& imagePath, const QString& branch);
    void sectorPreviewCompleted();
    void sectorPreviewFailed(const QString& errorText);

private slots:
    void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    QString workspaceRoot() const;
    QString toWslPath(const QString& windowsPath) const;
    QString fromWslPath(const QString& wslPath) const;
    QString shellQuote(const QString& value) const;
    QString safeStem(const QString& filePath) const;
    QVector<StepImage> collectStepImages(const AnalysisPayload& payload) const;
    bool parseStdoutPayload(const QByteArray& stdoutData, const QString& stderrText, AnalysisPayload& payload, QString& errorText) const;
    bool parseRoiStdoutPayload(const QByteArray& stdoutData, RoiOcrPayload& payload, QString& errorText) const;
    bool parseSectorStdoutPayload(const QByteArray& stdoutData, SectorPreviewPayload& payload, QString& errorText) const;

    QProcess process_;
    QProcess roiProcess_;
    QProcess sectorProcess_;
    AnalysisPayload lastPayload_;
    RoiOcrPayload lastRoiPayload_;
    SectorPreviewPayload lastSectorPayload_;
    QString lastError_;
};
