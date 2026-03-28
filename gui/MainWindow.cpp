#include "MainWindow.h"

#include <algorithm>

#include <QAction>
#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLineEdit>
#include <QImageReader>
#include <QMessageBox>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QStatusBar>
#include <QStyle>
#include <QToolBar>
#include <QToolButton>
#include <QVBoxLayout>

ImagePreviewLabel::ImagePreviewLabel(QWidget* parent) : QLabel(parent) {
    setAlignment(Qt::AlignCenter);
    setMinimumSize(320, 240);
    setText(QStringLiteral("Seleziona un'immagine dalla galleria."));
    setStyleSheet(QStringLiteral("QLabel { background: #151515; color: #f3f3f3; border: 1px solid #444; }"));
}

void ImagePreviewLabel::setImagePath(const QString& imagePath) {
    imagePath_ = imagePath;
    originalPixmap_ = QPixmap(imagePath_);
    if (originalPixmap_.isNull()) {
        setText(QStringLiteral("Impossibile caricare: %1").arg(imagePath));
        return;
    }
    refreshPixmap();
}

void ImagePreviewLabel::setZoomFactor(double zoomFactor) {
    zoomFactor_ = std::max(0.1, zoomFactor);
    refreshPixmap();
}

void ImagePreviewLabel::setFitToWindow(bool fitToWindow) {
    fitToWindow_ = fitToWindow;
    refreshPixmap();
}

double ImagePreviewLabel::zoomFactor() const {
    return zoomFactor_;
}

void ImagePreviewLabel::resizeEvent(QResizeEvent* event) {
    QLabel::resizeEvent(event);
    refreshPixmap();
}

void ImagePreviewLabel::refreshPixmap() {
    if (originalPixmap_.isNull()) {
        return;
    }
    if (fitToWindow_) {
        setPixmap(originalPixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        return;
    }

    const QSize scaledSize(
        std::max(1, static_cast<int>(originalPixmap_.width() * zoomFactor_)),
        std::max(1, static_cast<int>(originalPixmap_.height() * zoomFactor_)));
    const QPixmap scaledPixmap = originalPixmap_.scaled(scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    setPixmap(scaledPixmap);
    resize(scaledPixmap.size());
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    buildUi();
    connect(&backend_, &BackendClient::analysisStarted, this, &MainWindow::backendStarted);
    connect(&backend_, &BackendClient::analysisCompleted, this, &MainWindow::backendCompleted);
    connect(&backend_, &BackendClient::analysisFailed, this, &MainWindow::backendFailed);
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("Tyre Reader Debug GUI"));
    resize(1600, 980);

    auto* toolbar = addToolBar(QStringLiteral("Main"));
    toolbar->setMovable(false);
    toolbar->addAction(QStringLiteral("Apri cartella"), this, &MainWindow::openFolder);
    toolbar->addAction(QStringLiteral("Apri file"), this, &MainWindow::openFiles);

    auto* central = new QWidget(this);
    auto* mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(8, 8, 8, 8);
    mainLayout->setSpacing(8);

    currentImageLabel_ = new QLabel(QStringLiteral("Nessuna immagine caricata."), this);
    mainLayout->addWidget(currentImageLabel_);

    auto* controlsLayout = new QHBoxLayout();
    modeCombo_ = new QComboBox(this);
    modeCombo_->addItem(QStringLiteral("Analisi completa"), static_cast<int>(BackendClient::RunMode::FullAnalysis));
    modeCombo_->addItem(QStringLiteral("Solo geometria ruota"), static_cast<int>(BackendClient::RunMode::WheelGeometry));
    controlsLayout->addWidget(modeCombo_);

    analyzeButton_ = new QPushButton(QStringLiteral("Start elaborazione"), this);
    connect(analyzeButton_, &QPushButton::clicked, this, &MainWindow::analyzeCurrentSelection);
    controlsLayout->addWidget(analyzeButton_);

    galleryFilterEdit_ = new QLineEdit(this);
    galleryFilterEdit_->setPlaceholderText(QStringLiteral("Filtro gallery per nome file..."));
    connect(galleryFilterEdit_, &QLineEdit::textChanged, this, &MainWindow::applyGalleryFilter);
    controlsLayout->addWidget(galleryFilterEdit_, 1);

    fitButton_ = new QToolButton(this);
    fitButton_->setText(QStringLiteral("Adatta"));
    fitButton_->setCheckable(true);
    fitButton_->setChecked(true);
    connect(fitButton_, &QToolButton::toggled, this, &MainWindow::fitToggleChanged);
    controlsLayout->addWidget(fitButton_);

    zoomSlider_ = new QSlider(Qt::Horizontal, this);
    zoomSlider_->setRange(10, 400);
    zoomSlider_->setValue(100);
    zoomSlider_->setEnabled(false);
    zoomSlider_->setToolTip(QStringLiteral("Zoom immagine"));
    connect(zoomSlider_, &QSlider::valueChanged, this, &MainWindow::zoomSliderChanged);
    controlsLayout->addWidget(zoomSlider_);

    mainLayout->addLayout(controlsLayout);

    galleryList_ = new QListWidget(this);
    galleryList_->setViewMode(QListView::IconMode);
    galleryList_->setFlow(QListView::LeftToRight);
    galleryList_->setResizeMode(QListView::Adjust);
    galleryList_->setMovement(QListView::Static);
    galleryList_->setIconSize(QSize(128, 96));
    galleryList_->setWrapping(false);
    galleryList_->setMaximumHeight(140);
    galleryList_->setSpacing(8);
    connect(galleryList_, &QListWidget::currentItemChanged, this, &MainWindow::gallerySelectionChanged);
    mainLayout->addWidget(galleryList_);

    auto* bodySplitter = new QSplitter(Qt::Horizontal, this);
    auto* tabWidget = new QTabWidget(this);

    auto* imagesTab = new QWidget(this);
    auto* imagesLayout = new QHBoxLayout(imagesTab);
    imagesLayout->setContentsMargins(0, 0, 0, 0);
    imagesLayout->setSpacing(8);

    stepList_ = new QListWidget(this);
    stepList_->setMinimumWidth(280);
    connect(stepList_, &QListWidget::currentItemChanged, this, &MainWindow::stepSelectionChanged);
    imagesLayout->addWidget(stepList_);

    imagePreview_ = new ImagePreviewLabel(this);
    auto* previewScroll = new QScrollArea(this);
    previewScroll->setWidgetResizable(true);
    previewScroll->setBackgroundRole(QPalette::Dark);
    previewScroll->setWidget(imagePreview_);
    imagesLayout->addWidget(previewScroll, 1);

    tabWidget->addTab(imagesTab, QStringLiteral("Elaborazioni"));

    resultsTable_ = new QTableWidget(this);
    resultsTable_->setColumnCount(2);
    resultsTable_->setHorizontalHeaderLabels({QStringLiteral("Campo"), QStringLiteral("Valore")});
    resultsTable_->horizontalHeader()->setStretchLastSection(true);
    resultsTable_->verticalHeader()->setVisible(false);
    resultsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    resultsTable_->setSelectionMode(QAbstractItemView::NoSelection);
    tabWidget->addTab(resultsTable_, QStringLiteral("Risultati"));

    timingsTable_ = new QTableWidget(this);
    timingsTable_->setColumnCount(2);
    timingsTable_->setHorizontalHeaderLabels({QStringLiteral("Fase"), QStringLiteral("Tempo [ms]")});
    timingsTable_->horizontalHeader()->setStretchLastSection(true);
    timingsTable_->verticalHeader()->setVisible(false);
    timingsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    timingsTable_->setSelectionMode(QAbstractItemView::NoSelection);
    tabWidget->addTab(timingsTable_, QStringLiteral("Tempi"));

    logView_ = new QPlainTextEdit(this);
    logView_->setReadOnly(true);
    tabWidget->addTab(logView_, QStringLiteral("Log"));

    bodySplitter->addWidget(tabWidget);
    bodySplitter->setStretchFactor(0, 1);
    mainLayout->addWidget(bodySplitter, 1);

    setCentralWidget(central);

    statusLabel_ = new QLabel(QStringLiteral("Pronto."), this);
    statusBar()->addPermanentWidget(statusLabel_);
}

void MainWindow::openFolder() {
    const QString folder = QFileDialog::getExistingDirectory(this, QStringLiteral("Apri cartella immagini"));
    if (folder.isEmpty()) {
        return;
    }

    QDir dir(folder);
    QStringList filters;
    const auto formats = QImageReader::supportedImageFormats();
    for (const QByteArray& format : formats) {
        filters << QStringLiteral("*.%1").arg(QString::fromLatin1(format));
    }
    const QFileInfoList entries = dir.entryInfoList(filters, QDir::Files, QDir::Name | QDir::IgnoreCase);

    QStringList files;
    for (const QFileInfo& entry : entries) {
        files << entry.absoluteFilePath();
    }
    setImageList(files);
}

void MainWindow::openFiles() {
    QStringList filters;
    const auto formats = QImageReader::supportedImageFormats();
    for (const QByteArray& format : formats) {
        filters << QStringLiteral("*.%1").arg(QString::fromLatin1(format));
    }

    const QStringList files = QFileDialog::getOpenFileNames(
        this,
        QStringLiteral("Apri immagini"),
        QString(),
        QStringLiteral("Immagini (%1)").arg(filters.join(' ')));

    if (!files.isEmpty()) {
        setImageList(files);
    }
}

void MainWindow::setImageList(const QStringList& files) {
    imagePaths_ = files;
    galleryList_->clear();
    stepList_->clear();
    imagePreview_->setText(QStringLiteral("Seleziona una miniatura in alto."));
    imagePreview_->setFitToWindow(fitButton_->isChecked());
    logView_->clear();
    resultsTable_->setRowCount(0);
    timingsTable_->setRowCount(0);

    for (const QString& file : imagePaths_) {
        auto* item = new QListWidgetItem(QIcon(file), QFileInfo(file).fileName());
        item->setData(Qt::UserRole, file);
        galleryList_->addItem(item);
    }

    updateGalleryIcons();
    if (galleryList_->count() > 0) {
        galleryList_->setCurrentRow(0);
    } else {
        currentImageLabel_->setText(QStringLiteral("Nessuna immagine nella selezione."));
    }
    applyGalleryFilter(galleryFilterEdit_->text());
}

void MainWindow::updateGalleryIcons() {
    for (int row = 0; row < galleryList_->count(); ++row) {
        QListWidgetItem* item = galleryList_->item(row);
        const QString filePath = item->data(Qt::UserRole).toString();
        item->setIcon(QIcon(QPixmap(filePath).scaled(128, 96, Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    }
}

QString MainWindow::currentImagePath() const {
    if (QListWidgetItem* item = galleryList_->currentItem()) {
        return item->data(Qt::UserRole).toString();
    }
    return QString();
}

void MainWindow::gallerySelectionChanged() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    currentImageLabel_->setText(QStringLiteral("Selezionata: %1").arg(imagePath));
    stepList_->clear();
    imagePreview_->setImagePath(imagePath);
    resultsTable_->setRowCount(0);
    timingsTable_->setRowCount(0);
    logView_->setPlainText(QStringLiteral("Immagine selezionata. Premi 'Start elaborazione' per avviare il backend."));
}

void MainWindow::analyzeCurrentSelection() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    if (backend_.isBusy()) {
        return;
    }
    const auto mode = static_cast<BackendClient::RunMode>(modeCombo_->currentData().toInt());
    backend_.analyzeImage(imagePath, mode);
}

void MainWindow::applyGalleryFilter(const QString& filterText) {
    const QString normalizedFilter = filterText.trimmed();
    int firstVisibleRow = -1;
    for (int row = 0; row < galleryList_->count(); ++row) {
        QListWidgetItem* item = galleryList_->item(row);
        const bool matches = normalizedFilter.isEmpty() ||
            item->text().contains(normalizedFilter, Qt::CaseInsensitive);
        item->setHidden(!matches);
        if (matches && firstVisibleRow < 0) {
            firstVisibleRow = row;
        }
    }

    if (firstVisibleRow >= 0 && (galleryList_->currentItem() == nullptr || galleryList_->currentItem()->isHidden())) {
        galleryList_->setCurrentRow(firstVisibleRow);
    }
}

void MainWindow::backendStarted(const QString& imagePath) {
    statusLabel_->setText(QStringLiteral("Elaborazione in corso..."));
    analyzeButton_->setEnabled(false);
    modeCombo_->setEnabled(false);
    logView_->setPlainText(QStringLiteral("Backend avviato su:\n%1").arg(imagePath));
}

void MainWindow::backendCompleted() {
    const auto& payload = backend_.lastPayload();
    updateFromPayload(payload);
    statusLabel_->setText(QStringLiteral("Elaborazione completata."));
    analyzeButton_->setEnabled(true);
    modeCombo_->setEnabled(true);
}

void MainWindow::backendFailed(const QString& errorText) {
    statusLabel_->setText(QStringLiteral("Errore backend."));
    logView_->setPlainText(errorText);
    analyzeButton_->setEnabled(true);
    modeCombo_->setEnabled(true);
    QMessageBox::warning(this, QStringLiteral("Errore backend"), errorText);
}

void MainWindow::updateFromPayload(const BackendClient::AnalysisPayload& payload) {
    updateStepList(payload);
    updateResultsTable(payload);
    updateTimingsTable(payload);

    QStringList logLines;
    if (!payload.notes.isEmpty()) {
        logLines << QStringLiteral("Note:");
        for (const QString& note : payload.notes) {
            logLines << QStringLiteral(" - %1").arg(note);
        }
    }
    if (!payload.stderrText.isEmpty()) {
        logLines << QStringLiteral("") << QStringLiteral("stderr backend:") << payload.stderrText;
    }
    if (logLines.isEmpty()) {
        logLines << QStringLiteral("Nessuna nota speciale.");
    }
    logView_->setPlainText(logLines.join('\n'));

    if (stepList_->count() > 0) {
        stepList_->setCurrentRow(0);
    }
}

void MainWindow::updateResultsTable(const BackendClient::AnalysisPayload& payload) {
    QVector<QPair<QString, QString>> rows = {
        {QStringLiteral("Immagine"), payload.inputPath},
        {QStringLiteral("Frame ID"), payload.frameId},
        {QStringLiteral("Modalita'"), payload.mode == BackendClient::RunMode::WheelGeometry
            ? QStringLiteral("Solo geometria ruota")
            : QStringLiteral("Analisi completa")}
    };

    if (payload.mode == BackendClient::RunMode::WheelGeometry) {
        rows.push_back({QStringLiteral("Ruota trovata"), payload.wheelFound ? QStringLiteral("Si") : QStringLiteral("No")});
        rows.push_back({QStringLiteral("Originale copiato"), payload.originalCopyPath});
        rows.push_back({QStringLiteral("Overlay ruota"), payload.wheelOverlayPath});
        rows.push_back({QStringLiteral("Spalla linearizzata"), payload.unwrappedBandPath});
    } else {
        rows.push_back({QStringLiteral("Misura trovata"), payload.tyreSizeFound ? QStringLiteral("Si") : QStringLiteral("No")});
        rows.push_back({QStringLiteral("Misura normalizzata"), payload.tyreSize.normalized});
        rows.push_back({QStringLiteral("Misura raw"), payload.tyreSize.raw});
        rows.push_back({QStringLiteral("Confidenza misura"), QString::number(payload.tyreSize.confidence, 'f', 4)});
        rows.push_back({QStringLiteral("DOT trovato"), payload.dotFound ? QStringLiteral("Si") : QStringLiteral("No")});
        rows.push_back({QStringLiteral("DOT raw"), payload.dot.raw});
        rows.push_back({QStringLiteral("DOT normalizzato"), payload.dot.normalized});
        rows.push_back({QStringLiteral("DOT settimana/anno"), payload.dotWeekYear});
        rows.push_back({QStringLiteral("DOT completo"), payload.dotFullNormalized});
        rows.push_back({QStringLiteral("Overlay"), payload.overlayPath});
        rows.push_back({QStringLiteral("Debug dir"), payload.debugDir});
        rows.push_back({QStringLiteral("Timing report"), payload.timingReportPath});
        rows.push_back({QStringLiteral("OCR report"), payload.ocrReportPath});
    }

    resultsTable_->setRowCount(rows.size());
    for (int row = 0; row < rows.size(); ++row) {
        resultsTable_->setItem(row, 0, new QTableWidgetItem(rows[row].first));
        resultsTable_->setItem(row, 1, new QTableWidgetItem(rows[row].second));
    }
    resultsTable_->resizeColumnsToContents();
}

void MainWindow::updateTimingsTable(const BackendClient::AnalysisPayload& payload) {
    timingsTable_->setRowCount(payload.timings.size());
    for (int row = 0; row < payload.timings.size(); ++row) {
        timingsTable_->setItem(row, 0, new QTableWidgetItem(payload.timings[row].name));
        timingsTable_->setItem(row, 1, new QTableWidgetItem(QString::number(payload.timings[row].ms, 'f', 3)));
    }
    timingsTable_->resizeColumnsToContents();
}

void MainWindow::updateStepList(const BackendClient::AnalysisPayload& payload) {
    stepList_->clear();
    for (const auto& image : payload.images) {
        auto* item = new QListWidgetItem(image.label);
        item->setData(Qt::UserRole, image.path);
        stepList_->addItem(item);
    }
}

void MainWindow::stepSelectionChanged() {
    if (QListWidgetItem* item = stepList_->currentItem()) {
        imagePreview_->setImagePath(item->data(Qt::UserRole).toString());
    }
}

void MainWindow::zoomSliderChanged(int value) {
    if (fitButton_->isChecked()) {
        return;
    }
    imagePreview_->setZoomFactor(static_cast<double>(value) / 100.0);
}

void MainWindow::fitToggleChanged(bool checked) {
    zoomSlider_->setEnabled(!checked);
    imagePreview_->setFitToWindow(checked);
    if (!checked) {
        imagePreview_->setZoomFactor(static_cast<double>(zoomSlider_->value()) / 100.0);
    }
}
