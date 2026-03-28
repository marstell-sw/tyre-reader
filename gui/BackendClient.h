#pragma once

#include <QObject>
#include <QProcess>
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

    explicit BackendClient(QObject* parent = nullptr);

    bool isBusy() const;
    void analyzeImage(const QString& imagePath, RunMode mode);
    const AnalysisPayload& lastPayload() const;
    QString lastError() const;

signals:
    void analysisStarted(const QString& imagePath);
    void analysisCompleted();
    void analysisFailed(const QString& errorText);

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

    QProcess process_;
    AnalysisPayload lastPayload_;
    QString lastError_;
};
