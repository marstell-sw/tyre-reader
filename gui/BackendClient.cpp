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
    connect(&roiProcess_, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
        const QByteArray stdoutData = roiProcess_.readAllStandardOutput();
        const QString stderrText = QString::fromUtf8(roiProcess_.readAllStandardError()).trimmed();

        if (exitStatus != QProcess::NormalExit) {
            lastError_ = QStringLiteral("Il backend ROI OCR e' terminato in modo anomalo.");
            emit roiOcrFailed(lastError_);
            return;
        }

        QString errorText;
        if (exitCode != 0) {
            errorText = !stderrText.isEmpty() ? stderrText : QString::fromUtf8(stdoutData).trimmed();
            if (errorText.isEmpty()) {
                errorText = QStringLiteral("Il backend ROI OCR ha restituito exit code %1.").arg(exitCode);
            }
            lastError_ = errorText;
            emit roiOcrFailed(lastError_);
            return;
        }

        RoiOcrPayload payload;
        if (!parseRoiStdoutPayload(stdoutData, payload, errorText)) {
            lastError_ = errorText;
            emit roiOcrFailed(lastError_);
            return;
        }

        lastRoiPayload_ = payload;
        emit roiOcrCompleted();
    });
    connect(&sectorProcess_, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus exitStatus) {
        const QByteArray stdoutData = sectorProcess_.readAllStandardOutput();
        const QString stderrText = QString::fromUtf8(sectorProcess_.readAllStandardError()).trimmed();

        if (exitStatus != QProcess::NormalExit) {
            lastError_ = QStringLiteral("Il backend sector preview e' terminato in modo anomalo.");
            emit sectorPreviewFailed(lastError_);
            return;
        }

        QString errorText;
        if (exitCode != 0) {
            errorText = !stderrText.isEmpty() ? stderrText : QString::fromUtf8(stdoutData).trimmed();
            if (errorText.isEmpty()) {
                errorText = QStringLiteral("Il backend sector preview ha restituito exit code %1.").arg(exitCode);
            }
            lastError_ = errorText;
            emit sectorPreviewFailed(lastError_);
            return;
        }

        SectorPreviewPayload payload;
        if (!parseSectorStdoutPayload(stdoutData, payload, errorText)) {
            lastError_ = errorText;
            emit sectorPreviewFailed(lastError_);
            return;
        }

        lastSectorPayload_ = payload;
        emit sectorPreviewCompleted();
    });
}

bool BackendClient::isBusy() const {
    return process_.state() != QProcess::NotRunning;
}

bool BackendClient::isRoiBusy() const {
    return roiProcess_.state() != QProcess::NotRunning;
}

bool BackendClient::isSectorBusy() const {
    return sectorProcess_.state() != QProcess::NotRunning;
}

const BackendClient::AnalysisPayload& BackendClient::lastPayload() const {
    return lastPayload_;
}

const BackendClient::RoiOcrPayload& BackendClient::lastRoiPayload() const {
    return lastRoiPayload_;
}

const BackendClient::SectorPreviewPayload& BackendClient::lastSectorPayload() const {
    return lastSectorPayload_;
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

void BackendClient::runRoiOcr(const QString& imagePath, const QRect& roi, const QString& branch) {
    if (isRoiBusy()) {
        roiProcess_.kill();
        roiProcess_.waitForFinished(3000);
    }

    lastRoiPayload_ = RoiOcrPayload{};
    lastError_.clear();

    const QString runId = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    const QString outputDir = QDir(workspaceRoot()).filePath(QStringLiteral(".gui_debug_runs/roi_%1_%2").arg(safeStem(imagePath), runId));
    QDir().mkpath(outputDir);

    const QString workspaceWsl = toWslPath(workspaceRoot());
    const QString imageWsl = toWslPath(imagePath);
    const QString outputWsl = toWslPath(outputDir);
    const QString roiSpec = QStringLiteral("%1,%2,%3,%4").arg(roi.x()).arg(roi.y()).arg(roi.width()).arg(roi.height());
    const QString bashCommand =
        QStringLiteral("cd %1 && ./build-wsl24/tyre_reader_v3 --ocr-roi %2 --roi %3 --branch %4 --output %5 --pretty")
            .arg(shellQuote(workspaceWsl),
                 shellQuote(imageWsl),
                 shellQuote(roiSpec),
                 shellQuote(branch),
                 shellQuote(outputWsl));

    emit roiOcrStarted(imagePath, branch);
    roiProcess_.setProgram(QStringLiteral("wsl.exe"));
    roiProcess_.setArguments({
        QStringLiteral("-d"),
        QStringLiteral("Ubuntu-24.04"),
        QStringLiteral("bash"),
        QStringLiteral("-lc"),
        bashCommand
    });
    roiProcess_.start();
}

void BackendClient::runSectorPreview(const QString& imagePath,
                                     const QString& branch,
                                     double startAngleDeg,
                                     double endAngleDeg,
                                     bool useWheelOverride,
                                     double wheelCenterX,
                                     double wheelCenterY,
                                     double wheelInnerRadius,
                                     double wheelOuterRadius) {
    if (isSectorBusy()) {
        sectorProcess_.kill();
        sectorProcess_.waitForFinished(3000);
    }

    lastSectorPayload_ = SectorPreviewPayload{};
    lastError_.clear();

    const QString runId = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss_zzz"));
    const QString outputDir = QDir(workspaceRoot()).filePath(QStringLiteral(".gui_debug_runs/sector_%1_%2").arg(safeStem(imagePath), runId));
    QDir().mkpath(outputDir);

    const QString workspaceWsl = toWslPath(workspaceRoot());
    const QString imageWsl = toWslPath(imagePath);
    const QString outputWsl = toWslPath(outputDir);
    const QString angleSpec = QStringLiteral("%1,%2").arg(startAngleDeg, 0, 'f', 3).arg(endAngleDeg, 0, 'f', 3);
    const QString wheelOverrideSpec = QStringLiteral("%1,%2,%3,%4")
        .arg(wheelCenterX, 0, 'f', 3)
        .arg(wheelCenterY, 0, 'f', 3)
        .arg(wheelInnerRadius, 0, 'f', 3)
        .arg(wheelOuterRadius, 0, 'f', 3);
    QString bashCommand =
        QStringLiteral("cd %1 && ./build-wsl24/tyre_reader_v3 --unwrap-sector %2 --angles %3 --branch %4 --output %5 --pretty")
            .arg(shellQuote(workspaceWsl),
                 shellQuote(imageWsl),
                 shellQuote(angleSpec),
                 shellQuote(branch),
                 shellQuote(outputWsl));
    if (useWheelOverride) {
        bashCommand += QStringLiteral(" --wheel-override %1").arg(shellQuote(wheelOverrideSpec));
    }

    emit sectorPreviewStarted(imagePath, branch);
    sectorProcess_.setProgram(QStringLiteral("wsl.exe"));
    sectorProcess_.setArguments({
        QStringLiteral("-d"),
        QStringLiteral("Ubuntu-24.04"),
        QStringLiteral("bash"),
        QStringLiteral("-lc"),
        bashCommand
    });
    sectorProcess_.start();
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
    payload.wheelFound = root.value(QStringLiteral("wheelFound")).toBool();
    payload.wheelCenterX = root.value(QStringLiteral("wheelCenterX")).toDouble();
    payload.wheelCenterY = root.value(QStringLiteral("wheelCenterY")).toDouble();
    payload.wheelInnerRadius = root.value(QStringLiteral("wheelInnerRadius")).toDouble();
    payload.wheelOuterRadius = root.value(QStringLiteral("wheelOuterRadius")).toDouble();

    if (root.contains(QStringLiteral("originalCopyPath"))) {
        payload.mode = RunMode::WheelGeometry;
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

bool BackendClient::parseRoiStdoutPayload(const QByteArray& stdoutData, RoiOcrPayload& payload, QString& errorText) const {
    QJsonParseError parseError;
    const QJsonDocument json = QJsonDocument::fromJson(stdoutData, &parseError);
    if (parseError.error != QJsonParseError::NoError || !json.isObject()) {
        errorText = QStringLiteral("JSON ROI backend non valido: %1").arg(parseError.errorString());
        return false;
    }

    const QJsonObject root = json.object();
    if (root.contains(QStringLiteral("error"))) {
        errorText = root.value(QStringLiteral("error")).toString();
        return false;
    }

    payload.imagePath = fromWslPath(root.value(QStringLiteral("imagePath")).toString());
    payload.branch = root.value(QStringLiteral("branch")).toString();
    payload.rawText = root.value(QStringLiteral("rawText")).toString();
    payload.normalizedText = root.value(QStringLiteral("normalizedText")).toString();
    payload.found = root.value(QStringLiteral("found")).toBool();
    payload.confidence = root.value(QStringLiteral("confidence")).toDouble();
    payload.cropPath = fromWslPath(root.value(QStringLiteral("cropPath")).toString());

    const QString roiText = root.value(QStringLiteral("roi")).toString();
    const QStringList parts = roiText.split(',');
    if (parts.size() == 4) {
        payload.roi = QRect(parts[0].toInt(), parts[1].toInt(), parts[2].toInt(), parts[3].toInt());
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

    return true;
}

bool BackendClient::parseSectorStdoutPayload(const QByteArray& stdoutData, SectorPreviewPayload& payload, QString& errorText) const {
    QJsonParseError parseError;
    const QJsonDocument json = QJsonDocument::fromJson(stdoutData, &parseError);
    if (parseError.error != QJsonParseError::NoError || !json.isObject()) {
        errorText = QStringLiteral("JSON sector backend non valido: %1").arg(parseError.errorString());
        return false;
    }

    const QJsonObject root = json.object();
    if (root.contains(QStringLiteral("error"))) {
        errorText = root.value(QStringLiteral("error")).toString();
        return false;
    }

    payload.imagePath = fromWslPath(root.value(QStringLiteral("imagePath")).toString());
    payload.branch = root.value(QStringLiteral("branch")).toString();
    payload.startAngleDeg = root.value(QStringLiteral("startAngleDeg")).toDouble();
    payload.endAngleDeg = root.value(QStringLiteral("endAngleDeg")).toDouble();
    payload.wheelFound = root.value(QStringLiteral("wheelFound")).toBool();
    payload.wheelCenterX = root.value(QStringLiteral("wheelCenterX")).toDouble();
    payload.wheelCenterY = root.value(QStringLiteral("wheelCenterY")).toDouble();
    payload.wheelInnerRadius = root.value(QStringLiteral("wheelInnerRadius")).toDouble();
    payload.wheelOuterRadius = root.value(QStringLiteral("wheelOuterRadius")).toDouble();
    payload.overlayPath = fromWslPath(root.value(QStringLiteral("overlayPath")).toString());
    payload.unwrappedPath = fromWslPath(root.value(QStringLiteral("unwrappedPath")).toString());

    const QJsonArray notes = root.value(QStringLiteral("notes")).toArray();
    for (const QJsonValue& value : notes) {
        payload.notes.push_back(value.toString());
    }

    const QJsonArray stepTimings = root.value(QStringLiteral("stepTimings")).toArray();
    for (const QJsonValue& value : stepTimings) {
        const QJsonObject item = value.toObject();
        payload.timings.push_back({item.value(QStringLiteral("name")).toString(), item.value(QStringLiteral("ms")).toDouble()});
    }

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
