#include "MainWindow.h"

#include <algorithm>
#include <cmath>

#include <QAction>
#include <QApplication>
#include <QDir>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImageReader>
#include <QKeySequence>
#include <QMessageBox>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWheelEvent>

ImageViewer::ImageViewer(QWidget* parent) : QScrollArea(parent) {
    imageLabel_ = new QLabel(this);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setBackgroundRole(QPalette::Base);
    imageLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel_->installEventFilter(this);

    setWidget(imageLabel_);
    setWidgetResizable(false);
    setAlignment(Qt::AlignCenter);
    setBackgroundRole(QPalette::Dark);
    setFrameShape(QFrame::StyledPanel);
    setMinimumSize(320, 240);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    viewport()->installEventFilter(this);
    imageLabel_->setText(QStringLiteral("Seleziona un'immagine dalla lista a sinistra."));
}

void ImageViewer::setImagePath(const QString& imagePath) {
    imagePath_ = imagePath;
    originalPixmap_ = QPixmap(imagePath_);
    panning_ = false;

    if (originalPixmap_.isNull()) {
        imageLabel_->setPixmap(QPixmap());
        imageLabel_->setText(QStringLiteral("Impossibile caricare: %1").arg(imagePath));
        imageLabel_->adjustSize();
        return;
    }

    imageLabel_->setText(QString());
    refreshPixmap();
}

void ImageViewer::setZoomFactor(double zoomFactor) {
    zoomFactor_ = std::clamp(zoomFactor, 0.1, 8.0);
    if (!fitToWindow_) {
        refreshPixmap();
    }
}

void ImageViewer::setFitToWindow(bool fitToWindow) {
    fitToWindow_ = fitToWindow;
    refreshPixmap();
}

double ImageViewer::zoomFactor() const {
    return zoomFactor_;
}

bool ImageViewer::fitToWindow() const {
    return fitToWindow_;
}

void ImageViewer::zoomIn() {
    fitToWindow_ = false;
    setZoomFactor(zoomFactor_ * 1.15);
}

void ImageViewer::zoomOut() {
    fitToWindow_ = false;
    setZoomFactor(zoomFactor_ / 1.15);
}

void ImageViewer::resetView() {
    zoomFactor_ = 1.0;
    fitToWindow_ = true;
    refreshPixmap();
}

bool ImageViewer::eventFilter(QObject* watched, QEvent* event) {
    if (watched == imageLabel_ || watched == viewport()) {
        if (event->type() == QEvent::Wheel) {
            auto* wheelEvent = static_cast<QWheelEvent*>(event);
            if (wheelEvent->modifiers().testFlag(Qt::ControlModifier)) {
                if (wheelEvent->angleDelta().y() > 0) {
                    zoomIn();
                } else if (wheelEvent->angleDelta().y() < 0) {
                    zoomOut();
                }
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonPress) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton && mouseEvent->modifiers().testFlag(Qt::ControlModifier) && !fitToWindow_) {
                panning_ = true;
                lastMousePos_ = mouseEvent->globalPosition().toPoint();
                viewport()->setCursor(Qt::ClosedHandCursor);
                return true;
            }
        } else if (event->type() == QEvent::MouseMove) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (panning_) {
                const QPoint currentPos = mouseEvent->globalPosition().toPoint();
                applyPanDelta(currentPos - lastMousePos_);
                lastMousePos_ = currentPos;
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonRelease) {
            auto* mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton && panning_) {
                panning_ = false;
                viewport()->unsetCursor();
                return true;
            }
        }
    }

    return QScrollArea::eventFilter(watched, event);
}

void ImageViewer::resizeEvent(QResizeEvent* event) {
    QScrollArea::resizeEvent(event);
    if (fitToWindow_) {
        refreshPixmap();
    }
}

void ImageViewer::refreshPixmap() {
    if (originalPixmap_.isNull()) {
        return;
    }

    if (fitToWindow_) {
        const QSize targetSize = viewport()->size().boundedTo(originalPixmap_.size());
        const QPixmap scaled = originalPixmap_.scaled(
            targetSize.isValid() ? targetSize : viewport()->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
        imageLabel_->setPixmap(scaled);
        imageLabel_->resize(scaled.size());
        imageLabel_->adjustSize();
        return;
    }

    const QSize scaledSize(
        std::max(1, static_cast<int>(originalPixmap_.width() * zoomFactor_)),
        std::max(1, static_cast<int>(originalPixmap_.height() * zoomFactor_)));
    const QPixmap scaled = originalPixmap_.scaled(scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    imageLabel_->setPixmap(scaled);
    imageLabel_->resize(scaled.size());
    imageLabel_->adjustSize();
}

void ImageViewer::applyPanDelta(const QPoint& delta) {
    horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
    verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    buildUi();
    connect(&backend_, &BackendClient::analysisStarted, this, &MainWindow::backendStarted);
    connect(&backend_, &BackendClient::analysisCompleted, this, &MainWindow::backendCompleted);
    connect(&backend_, &BackendClient::analysisFailed, this, &MainWindow::backendFailed);
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("Tyre Reader Debug GUI"));
    resize(1680, 1000);

    auto* toolbar = addToolBar(QStringLiteral("Main"));
    toolbar->setMovable(false);
    toolbar->addAction(QStringLiteral("Apri cartella"), this, &MainWindow::openFolder);
    toolbar->addAction(QStringLiteral("Apri file"), this, &MainWindow::openFiles);
    toolbar->addSeparator();
    QAction* zoomInAction = toolbar->addAction(QStringLiteral("Zoom +"), this, &MainWindow::zoomIn);
    zoomInAction->setShortcut(QKeySequence(QStringLiteral("Ctrl++")));
    addAction(zoomInAction);
    QAction* zoomOutAction = toolbar->addAction(QStringLiteral("Zoom -"), this, &MainWindow::zoomOut);
    zoomOutAction->setShortcut(QKeySequence(QStringLiteral("Ctrl+-")));
    addAction(zoomOutAction);
    QAction* resetZoomAction = toolbar->addAction(QStringLiteral("Reset vista"), this, &MainWindow::resetZoom);
    resetZoomAction->setShortcut(QKeySequence(QStringLiteral("Ctrl+0")));
    addAction(resetZoomAction);

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
    zoomSlider_->setToolTip(QStringLiteral("Zoom [%]"));
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
    galleryList_->setMaximumHeight(150);
    galleryList_->setSpacing(8);
    galleryList_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    galleryList_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
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

    imageViewer_ = new ImageViewer(this);
    imagesLayout->addWidget(imageViewer_, 1);

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

    QStringList files;
    const QFileInfoList entries = dir.entryInfoList(filters, QDir::Files, QDir::Name | QDir::IgnoreCase);
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
    imageViewer_->setImagePath(QString());
    imageViewer_->resetView();
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
    syncZoomControlsFromViewer();
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
    imageViewer_->setImagePath(imagePath);
    imageViewer_->resetView();
    resultsTable_->setRowCount(0);
    timingsTable_->setRowCount(0);
    logView_->setPlainText(QStringLiteral("Immagine selezionata. Premi 'Start elaborazione' per avviare il backend."));
    syncZoomControlsFromViewer();
}

void MainWindow::applyGalleryFilter(const QString& filterText) {
    const QString normalizedFilter = filterText.trimmed();
    int firstVisibleRow = -1;
    for (int row = 0; row < galleryList_->count(); ++row) {
        QListWidgetItem* item = galleryList_->item(row);
        const bool matches = normalizedFilter.isEmpty() || item->text().contains(normalizedFilter, Qt::CaseInsensitive);
        item->setHidden(!matches);
        if (matches && firstVisibleRow < 0) {
            firstVisibleRow = row;
        }
    }

    if (firstVisibleRow >= 0) {
        QListWidgetItem* current = galleryList_->currentItem();
        if (current == nullptr || current->isHidden()) {
            galleryList_->setCurrentRow(firstVisibleRow);
        }
    }
}

void MainWindow::analyzeCurrentSelection() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty() || backend_.isBusy()) {
        return;
    }

    const auto mode = static_cast<BackendClient::RunMode>(modeCombo_->currentData().toInt());
    backend_.analyzeImage(imagePath, mode);
}

void MainWindow::backendStarted(const QString& imagePath) {
    statusLabel_->setText(QStringLiteral("Elaborazione in corso..."));
    analyzeButton_->setEnabled(false);
    modeCombo_->setEnabled(false);
    logView_->setPlainText(QStringLiteral("Backend avviato su:\n%1").arg(imagePath));
}

void MainWindow::backendCompleted() {
    updateFromPayload(backend_.lastPayload());
    statusLabel_->setText(QStringLiteral("Elaborazione completata."));
    analyzeButton_->setEnabled(true);
    modeCombo_->setEnabled(true);
}

void MainWindow::backendFailed(const QString& errorText) {
    statusLabel_->setText(QStringLiteral("Errore backend."));
    analyzeButton_->setEnabled(true);
    modeCombo_->setEnabled(true);
    logView_->setPlainText(errorText);
    QMessageBox::warning(this, QStringLiteral("Errore backend"), errorText);
}

void MainWindow::stepSelectionChanged() {
    if (QListWidgetItem* item = stepList_->currentItem()) {
        imageViewer_->setImagePath(item->data(Qt::UserRole).toString());
        syncZoomControlsFromViewer();
    }
}

void MainWindow::zoomSliderChanged(int value) {
    if (fitButton_->isChecked()) {
        return;
    }
    imageViewer_->setZoomFactor(static_cast<double>(value) / 100.0);
}

void MainWindow::fitToggleChanged(bool checked) {
    zoomSlider_->setEnabled(!checked);
    imageViewer_->setFitToWindow(checked);
    syncZoomControlsFromViewer();
}

void MainWindow::zoomIn() {
    fitButton_->setChecked(false);
    imageViewer_->zoomIn();
    syncZoomControlsFromViewer();
}

void MainWindow::zoomOut() {
    fitButton_->setChecked(false);
    imageViewer_->zoomOut();
    syncZoomControlsFromViewer();
}

void MainWindow::resetZoom() {
    imageViewer_->resetView();
    syncZoomControlsFromViewer();
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

void MainWindow::syncZoomControlsFromViewer() {
    const QSignalBlocker blockerSlider(zoomSlider_);
    const QSignalBlocker blockerButton(fitButton_);
    fitButton_->setChecked(imageViewer_->fitToWindow());
    zoomSlider_->setEnabled(!imageViewer_->fitToWindow());
    zoomSlider_->setValue(static_cast<int>(std::round(imageViewer_->zoomFactor() * 100.0)));
}
