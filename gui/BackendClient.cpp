#include "BackendClient.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace {

QString stepLabelFromPath(const QString& path) {
    return QFileInfo(path).completeBaseName();
}

}  // namespace

BackendClient::BackendClient(QObject* parent) : QObject(parent) {
    connect(&process_, &QProcess::finished, this, &BackendClient::handleProcessFinished);
}

bool BackendClient::isBusy() const {
    return process_.state() != QProcess::NotRunning;
}

const BackendClient::AnalysisPayload& BackendClient::lastPayload() const {
    return lastPayload_;
}

QString BackendClient::lastError() const {
    return lastError_;
}

QString BackendClient::workspaceRoot() const {
    return QString::fromUtf8(TYRE_READER_REPO_ROOT);
}

QString BackendClient::toWslPath(const QString& windowsPath) const {
    const QString normalized = QDir::toNativeSeparators(QFileInfo(windowsPath).absoluteFilePath());
    if (normalized.size() >= 2 && normalized[1] == QChar(':')) {
        const QChar drive = normalized[0].toLower();
        QString tail = normalized.mid(2);
        tail.replace('\\', '/');
        return QStringLiteral("/mnt/%1%2").arg(drive, tail);
    }
    return normalized;
}

QString BackendClient::fromWslPath(const QString& wslPath) const {
    if (wslPath.startsWith(QStringLiteral("/mnt/")) && wslPath.size() > 6) {
        const QChar drive = wslPath[5].toUpper();
        QString tail = wslPath.mid(6);
        tail.replace('/', '\\');
        return QStringLiteral("%1:%2").arg(drive, tail);
    }
    return wslPath;
}

QString BackendClient::shellQuote(const QString& value) const {
    QString escaped = value;
    escaped.replace('\'', QStringLiteral("'\"'\"'"));
    return QStringLiteral("'%1'").arg(escaped);
}

QString BackendClient::safeStem(const QString& filePath) const {
    QString stem = QFileInfo(filePath).completeBaseName();
    for (QChar& ch : stem) {
        if (!ch.isLetterOrNumber()) {
            ch = '_';
        }
    }
    return stem;
}

void BackendClient::analyzeImage(const QString& imagePath, RunMode mode) {
    if (isBusy()) {
        process_.kill();
        process_.waitForFinished(3000);
    }

    lastPayload_ = AnalysisPayload{};
    lastError_.clear();

    const QString runId = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    const QString outputDir = QDir(workspaceRoot()).filePath(QStringLiteral(".gui_debug_runs/%1_%2").arg(safeStem(imagePath), runId));
    QDir().mkpath(outputDir);

    const QString workspaceWsl = toWslPath(workspaceRoot());
    const QString imageWsl = toWslPath(imagePath);
    const QString outputWsl = toWslPath(outputDir);
    const QString cliArgs = (mode == RunMode::WheelGeometry)
        ? QStringLiteral("--wheel-image %1 --output %2 --pretty").arg(shellQuote(imageWsl), shellQuote(outputWsl))
        : QStringLiteral("--image %1 --output %2 --debug-steps --pretty").arg(shellQuote(imageWsl), shellQuote(outputWsl));
    const QString bashCommand =
        QStringLiteral("cd %1 && ./build-wsl24/tyre_reader_v3 %2").arg(shellQuote(workspaceWsl), cliArgs);

    emit analysisStarted(imagePath);
    process_.setProgram(QStringLiteral("wsl.exe"));
    process_.setArguments({
        QStringLiteral("-d"),
        QStringLiteral("Ubuntu-24.04"),
        QStringLiteral("bash"),
        QStringLiteral("-lc"),
        bashCommand
    });
    process_.start();
}

QVector<BackendClient::StepImage> BackendClient::collectStepImages(const AnalysisPayload& payload) const {
    QVector<StepImage> images;
    if (!payload.originalCopyPath.isEmpty() && QFileInfo::exists(payload.originalCopyPath)) {
        images.push_back({QStringLiteral("Originale"), payload.originalCopyPath});
    } else if (!payload.inputPath.isEmpty() && QFileInfo::exists(payload.inputPath)) {
        images.push_back({QStringLiteral("Originale"), payload.inputPath});
    }

    if (!payload.wheelOverlayPath.isEmpty() && QFileInfo::exists(payload.wheelOverlayPath)) {
        images.push_back({QStringLiteral("Ruota evidenziata"), payload.wheelOverlayPath});
    }
    if (!payload.unwrappedBandPath.isEmpty() && QFileInfo::exists(payload.unwrappedBandPath)) {
        images.push_back({QStringLiteral("Spalla linearizzata"), payload.unwrappedBandPath});
    }
    if (!payload.overlayPath.isEmpty() && QFileInfo::exists(payload.overlayPath)) {
        images.push_back({QStringLiteral("Overlay finale"), payload.overlayPath});
    }

    if (!payload.debugDir.isEmpty()) {
        QDir debugDir(payload.debugDir);
        const QFileInfoList files = debugDir.entryInfoList(
            {QStringLiteral("*.png"), QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"), QStringLiteral("*.bmp")},
            QDir::Files,
            QDir::Name | QDir::IgnoreCase);

        for (const QFileInfo& fileInfo : files) {
            images.push_back({stepLabelFromPath(fileInfo.fileName()), fileInfo.absoluteFilePath()});
        }
    }
    return images;
}

bool BackendClient::parseStdoutPayload(const QByteArray& stdoutData,
                                       const QString& stderrText,
                                       AnalysisPayload& payload,
                                       QString& errorText) const {
    QJsonParseError parseError;
    const QJsonDocument json = QJsonDocument::fromJson(stdoutData, &parseError);
    if (parseError.error != QJsonParseError::NoError || !json.isObject()) {
        errorText = QStringLiteral("JSON backend non valido: %1").arg(parseError.errorString());
        return false;
    }

    const QJsonObject root = json.object();
    if (root.contains(QStringLiteral("error"))) {
        errorText = root.value(QStringLiteral("error")).toString();
        return false;
    }

    const auto parseField = [this](const QJsonObject& obj) {
        ResultField field;
        field.raw = obj.value(QStringLiteral("raw")).toString();
        field.normalized = obj.value(QStringLiteral("normalized")).toString();
        field.found = obj.value(QStringLiteral("found")).toBool();
        field.confidence = obj.value(QStringLiteral("confidence")).toDouble();
        field.cropPath = fromWslPath(obj.value(QStringLiteral("cropPath")).toString());
        return field;
    };

    payload.inputPath = fromWslPath(root.value(QStringLiteral("inputPath")).toString());
    payload.frameId = root.value(QStringLiteral("frameId")).toString();
    payload.stderrText = stderrText;

    if (root.contains(QStringLiteral("wheelFound"))) {
        payload.mode = RunMode::WheelGeometry;
        payload.wheelFound = root.value(QStringLiteral("wheelFound")).toBool();
        payload.originalCopyPath = fromWslPath(root.value(QStringLiteral("originalCopyPath")).toString());
        payload.wheelOverlayPath = fromWslPath(root.value(QStringLiteral("wheelOverlayPath")).toString());
        payload.unwrappedBandPath = fromWslPath(root.value(QStringLiteral("unwrappedBandPath")).toString());
    } else {
        payload.mode = RunMode::FullAnalysis;
        payload.overlayPath = fromWslPath(root.value(QStringLiteral("overlayPath")).toString());
        payload.debugDir = fromWslPath(root.value(QStringLiteral("debugDir")).toString());
        payload.timingReportPath = fromWslPath(root.value(QStringLiteral("timingReportPath")).toString());
        payload.ocrReportPath = fromWslPath(root.value(QStringLiteral("ocrReportPath")).toString());
        payload.tyreSize = parseField(root.value(QStringLiteral("tyreSize")).toObject());
        payload.dot = parseField(root.value(QStringLiteral("dot")).toObject());
        payload.tyreSizeFound = root.value(QStringLiteral("tyreSizeFound")).toBool();
        payload.dotFound = root.value(QStringLiteral("dotFound")).toBool();
        payload.dotWeekYearFound = root.value(QStringLiteral("dotWeekYearFound")).toBool();
        payload.dotFullFound = root.value(QStringLiteral("dotFullFound")).toBool();
        payload.dotWeekYear = root.value(QStringLiteral("dotWeekYear")).toString();
        payload.dotFullRaw = root.value(QStringLiteral("dotFullRaw")).toString();
        payload.dotFullNormalized = root.value(QStringLiteral("dotFullNormalized")).toString();
    }

    const QJsonArray notes = root.value(QStringLiteral("notes")).toArray();
    for (const QJsonValue& value : notes) {
        payload.notes.push_back(value.toString());
    }

    const QJsonArray stepTimings = root.value(QStringLiteral("stepTimings")).toArray();
    for (const QJsonValue& value : stepTimings) {
        const QJsonObject item = value.toObject();
        payload.timings.push_back({item.value(QStringLiteral("name")).toString(), item.value(QStringLiteral("ms")).toDouble()});
    }

    payload.images = collectStepImages(payload);
    return true;
}

void BackendClient::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    const QByteArray stdoutData = process_.readAllStandardOutput();
    const QString stderrText = QString::fromUtf8(process_.readAllStandardError()).trimmed();

    if (exitStatus != QProcess::NormalExit) {
        lastError_ = QStringLiteral("Il backend e' terminato in modo anomalo.");
        emit analysisFailed(lastError_);
        return;
    }

    AnalysisPayload payload;
    QString errorText;
    if (exitCode != 0) {
        errorText = stderrText;
        if (errorText.isEmpty()) {
            errorText = QString::fromUtf8(stdoutData).trimmed();
        }
        if (errorText.isEmpty()) {
            errorText = QStringLiteral("Il backend ha restituito exit code %1.").arg(exitCode);
        }
        lastError_ = errorText;
        emit analysisFailed(lastError_);
        return;
    }

    if (!parseStdoutPayload(stdoutData, stderrText, payload, errorText)) {
        lastError_ = errorText;
        emit analysisFailed(lastError_);
        return;
    }

    lastPayload_ = payload;
    emit analysisCompleted();
}
