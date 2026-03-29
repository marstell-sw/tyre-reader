#include "MainWindow.h"

#include <algorithm>
#include <cmath>

#include <QAction>
#include <QApplication>
#include <QDir>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QImage>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeySequence>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QTextStream>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWheelEvent>

namespace {

QStringList parseCsvRow(const QString& line) {
    QStringList fields;
    QString current;
    bool inQuotes = false;
    for (int i = 0; i < line.size(); ++i) {
        const QChar ch = line.at(i);
        if (ch == '"') {
            if (inQuotes && i + 1 < line.size() && line.at(i + 1) == '"') {
                current += '"';
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch == ',' && !inQuotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current += ch;
        }
    }
    fields.push_back(current);
    return fields;
}

QString csvField(const QStringList& row, int index) {
    if (index < 0 || index >= row.size()) {
        return QString();
    }
    return row.at(index).trimmed();
}

}

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
    if (imagePath_.trimmed().isEmpty()) {
        originalPixmap_ = QPixmap();
        imageLabel_->setPixmap(QPixmap());
        imageLabel_->setText(QStringLiteral("Nessuna immagine disponibile."));
        imageLabel_->adjustSize();
        return;
    }
    originalPixmap_ = QPixmap(imagePath_);
    panning_ = false;

    if (originalPixmap_.isNull()) {
        imageLabel_->setPixmap(QPixmap());
        imageLabel_->setText(QStringLiteral("Impossibile caricare: %1").arg(QDir::toNativeSeparators(imagePath)));
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

void ImageViewer::setAnnotationBoxes(const QVector<AnnotationBox>& boxes) {
    annotationBoxes_ = boxes;
    refreshPixmap();
}

void ImageViewer::setSectorAnnotations(const QVector<SectorAnnotation>& sectors) {
    sectorAnnotations_ = sectors;
    refreshPixmap();
}

void ImageViewer::setInteractionMode(InteractionMode mode) {
    interactionMode_ = mode;
    drawingAnnotation_ = false;
    viewport()->unsetCursor();
    if (interactionMode_ == InteractionMode::View) {
        imageLabel_->setCursor(Qt::ArrowCursor);
    } else {
        imageLabel_->setCursor(Qt::CrossCursor);
    }
}

void ImageViewer::setWheelGeometry(bool found, const QPointF& center, double innerRadius, double outerRadius) {
    wheelGeometryFound_ = found;
    wheelCenter_ = center;
    wheelInnerRadius_ = innerRadius;
    wheelOuterRadius_ = outerRadius;
    refreshPixmap();
}

bool ImageViewer::eventFilter(QObject* watched, QEvent* event) {
    if (watched == imageLabel_ || watched == viewport()) {
        if (watched == imageLabel_ && interactionMode_ != InteractionMode::View) {
            if (event->type() == QEvent::MouseButtonPress) {
                auto* mouseEvent = static_cast<QMouseEvent*>(event);
                if (mouseEvent->button() == Qt::LeftButton && !originalPixmap_.isNull()) {
                    drawingAnnotation_ = true;
                    annotationStartImage_ = mapLabelPointToImage(mouseEvent->position().toPoint());
                    annotationEndImage_ = annotationStartImage_;
                    refreshPixmap();
                    return true;
                }
            } else if (event->type() == QEvent::MouseMove && drawingAnnotation_) {
                auto* mouseEvent = static_cast<QMouseEvent*>(event);
                annotationEndImage_ = mapLabelPointToImage(mouseEvent->position().toPoint());
                refreshPixmap();
                return true;
            } else if (event->type() == QEvent::MouseButtonRelease && drawingAnnotation_) {
                auto* mouseEvent = static_cast<QMouseEvent*>(event);
                if (mouseEvent->button() == Qt::LeftButton) {
                    drawingAnnotation_ = false;
                    annotationEndImage_ = mapLabelPointToImage(mouseEvent->position().toPoint());
                    if (wheelGeometryFound_) {
                        const double startAngle = angleForImagePoint(annotationStartImage_);
                        const double endAngle = angleForImagePoint(annotationEndImage_);
                        const QString label = interactionMode_ == InteractionMode::DrawSizeSector ? QStringLiteral("SIZE") : QStringLiteral("DOT");
                        emit sectorDrawn(label, startAngle, endAngle);
                    } else {
                        QRect imageRect(annotationStartImage_, annotationEndImage_);
                        imageRect = imageRect.normalized();
                        if (imageRect.width() >= 8 && imageRect.height() >= 8) {
                            const QString label = interactionMode_ == InteractionMode::DrawSizeSector ? QStringLiteral("SIZE") : QStringLiteral("DOT");
                            emit annotationDrawn(label, imageRect);
                        }
                    }
                    refreshPixmap();
                    return true;
                }
            }
        }

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

    QPixmap scaled;
    if (fitToWindow_) {
        const QSize targetSize = viewport()->size().boundedTo(originalPixmap_.size());
        scaled = originalPixmap_.scaled(
            targetSize.isValid() ? targetSize : viewport()->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
    } else {
        const QSize scaledSize(
            std::max(1, static_cast<int>(originalPixmap_.width() * zoomFactor_)),
            std::max(1, static_cast<int>(originalPixmap_.height() * zoomFactor_)));
        scaled = originalPixmap_.scaled(scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    if (!annotationBoxes_.isEmpty() || !sectorAnnotations_.isEmpty() || drawingAnnotation_ || wheelGeometryFound_) {
        QPixmap annotated = scaled;
        QPainter painter(&annotated);
        const double sx = static_cast<double>(annotated.width()) / std::max(1, originalPixmap_.width());
        const double sy = static_cast<double>(annotated.height()) / std::max(1, originalPixmap_.height());
        painter.setRenderHint(QPainter::Antialiasing, true);
        if (wheelGeometryFound_) {
            painter.setPen(QPen(QColor(0, 255, 0), 2));
            painter.drawEllipse(QPointF(wheelCenter_.x() * sx, wheelCenter_.y() * sy), wheelOuterRadius_ * sx, wheelOuterRadius_ * sy);
            painter.setPen(QPen(QColor(255, 140, 0), 2));
            painter.drawEllipse(QPointF(wheelCenter_.x() * sx, wheelCenter_.y() * sy), wheelInnerRadius_ * sx, wheelInnerRadius_ * sy);
        }
        for (const auto& box : annotationBoxes_) {
            QRect drawRect(
                static_cast<int>(std::round(box.imageRect.x() * sx)),
                static_cast<int>(std::round(box.imageRect.y() * sy)),
                static_cast<int>(std::round(box.imageRect.width() * sx)),
                static_cast<int>(std::round(box.imageRect.height() * sy)));
            painter.setPen(QPen(box.color, 2));
            painter.drawRect(drawRect);
            painter.drawText(drawRect.topLeft() + QPoint(0, -4), box.label);
        }
        const auto drawSector = [&](const SectorAnnotation& sector, Qt::PenStyle style) {
            if (!wheelGeometryFound_) {
                return;
            }
            painter.setPen(QPen(sector.color, 3, style));
            const auto drawRadius = [&](double angleDeg) {
                const double angleRad = angleDeg * M_PI / 180.0;
                const QPointF p0((wheelCenter_.x() + wheelInnerRadius_ * std::cos(angleRad)) * sx,
                                 (wheelCenter_.y() + wheelInnerRadius_ * std::sin(angleRad)) * sy);
                const QPointF p1((wheelCenter_.x() + wheelOuterRadius_ * std::cos(angleRad)) * sx,
                                 (wheelCenter_.y() + wheelOuterRadius_ * std::sin(angleRad)) * sy);
                painter.drawLine(p0, p1);
            };
            drawRadius(sector.startAngleDeg);
            drawRadius(sector.endAngleDeg);
            const QRectF outerRect((wheelCenter_.x() - wheelOuterRadius_) * sx,
                                   (wheelCenter_.y() - wheelOuterRadius_) * sy,
                                   2.0 * wheelOuterRadius_ * sx,
                                   2.0 * wheelOuterRadius_ * sy);
            double span = sector.endAngleDeg - sector.startAngleDeg;
            if (span <= 0.0) {
                span += 360.0;
            }
            painter.drawArc(outerRect, static_cast<int>(-sector.startAngleDeg * 16.0), static_cast<int>(-span * 16.0));
            painter.drawText(QPointF((wheelCenter_.x() + wheelOuterRadius_ + 8.0) * sx, (wheelCenter_.y() - 8.0) * sy), sector.label);
        };
        for (const auto& sector : sectorAnnotations_) {
            drawSector(sector, Qt::SolidLine);
        }
        if (drawingAnnotation_) {
            if (wheelGeometryFound_) {
                SectorAnnotation preview;
                preview.label = interactionMode_ == InteractionMode::DrawSizeSector ? QStringLiteral("SIZE") : QStringLiteral("DOT");
                preview.color = Qt::yellow;
                preview.startAngleDeg = angleForImagePoint(annotationStartImage_);
                preview.endAngleDeg = angleForImagePoint(annotationEndImage_);
                drawSector(preview, Qt::DashLine);
            } else {
                QRect current(annotationStartImage_, annotationEndImage_);
                current = current.normalized();
                QRect drawRect(
                    static_cast<int>(std::round(current.x() * sx)),
                    static_cast<int>(std::round(current.y() * sy)),
                    static_cast<int>(std::round(current.width() * sx)),
                    static_cast<int>(std::round(current.height() * sy)));
                painter.setPen(QPen(Qt::yellow, 2, Qt::DashLine));
                painter.drawRect(drawRect);
            }
        }
        painter.end();
        scaled = annotated;
    }

    imageLabel_->setPixmap(scaled);
    imageLabel_->resize(scaled.size());
    imageLabel_->adjustSize();
}

void ImageViewer::applyPanDelta(const QPoint& delta) {
    horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
    verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
}

QRect ImageViewer::displayedImageRect() const {
    if (imageLabel_ == nullptr || imageLabel_->pixmap().isNull()) {
        return {};
    }
    return QRect(QPoint(0, 0), imageLabel_->pixmap().size());
}

QPoint ImageViewer::mapLabelPointToImage(const QPoint& labelPoint) const {
    const QRect displayRect = displayedImageRect();
    if (displayRect.isEmpty() || originalPixmap_.isNull()) {
        return {};
    }
    const QPoint clamped(
        std::clamp(labelPoint.x(), displayRect.left(), displayRect.right()),
        std::clamp(labelPoint.y(), displayRect.top(), displayRect.bottom()));
    const double sx = static_cast<double>(originalPixmap_.width()) / std::max(1, displayRect.width());
    const double sy = static_cast<double>(originalPixmap_.height()) / std::max(1, displayRect.height());
    return QPoint(
        std::clamp(static_cast<int>(std::round(clamped.x() * sx)), 0, originalPixmap_.width() - 1),
        std::clamp(static_cast<int>(std::round(clamped.y() * sy)), 0, originalPixmap_.height() - 1));
}

double ImageViewer::angleForImagePoint(const QPoint& imagePoint) const {
    const double dx = static_cast<double>(imagePoint.x()) - wheelCenter_.x();
    const double dy = static_cast<double>(imagePoint.y()) - wheelCenter_.y();
    double angleDeg = std::atan2(dy, dx) * 180.0 / M_PI;
    if (angleDeg < 0.0) {
        angleDeg += 360.0;
    }
    return angleDeg;
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    buildUi();
    connect(&backend_, &BackendClient::analysisStarted, this, &MainWindow::backendStarted);
    connect(&backend_, &BackendClient::analysisCompleted, this, &MainWindow::backendCompleted);
    connect(&backend_, &BackendClient::analysisFailed, this, &MainWindow::backendFailed);
    connect(annotationViewer_, &ImageViewer::annotationDrawn, this, &MainWindow::handleAnnotationDrawn);
    connect(annotationViewer_, &ImageViewer::sectorDrawn, this, &MainWindow::handleSectorDrawn);
    connect(&backend_, &BackendClient::roiOcrStarted, this, [this](const QString&, const QString& branch) {
        statusLabel_->setText(QStringLiteral("OCR ROI in corso (%1)...").arg(branch.toUpper()));
        if (ocrSizeButton_ != nullptr) {
            ocrSizeButton_->setEnabled(false);
        }
        if (ocrDotButton_ != nullptr) {
            ocrDotButton_->setEnabled(false);
        }
    });
    connect(&backend_, &BackendClient::roiOcrCompleted, this, [this]() {
        const auto& payload = backend_.lastRoiPayload();
        QStringList lines;
        lines << QStringLiteral("ROI OCR %1").arg(payload.branch.toUpper());
        lines << QStringLiteral("ROI: %1,%2,%3,%4").arg(payload.roi.x()).arg(payload.roi.y()).arg(payload.roi.width()).arg(payload.roi.height());
        lines << QStringLiteral("Found: %1").arg(payload.found ? QStringLiteral("Si") : QStringLiteral("No"));
        lines << QStringLiteral("Raw: %1").arg(payload.rawText);
        lines << QStringLiteral("Normalized: %1").arg(payload.normalizedText);
        lines << QStringLiteral("Confidence: %1").arg(QString::number(payload.confidence, 'f', 4));
        for (const auto& timing : payload.timings) {
            lines << QStringLiteral("%1: %2 ms").arg(timing.name).arg(QString::number(timing.ms, 'f', 3));
        }
        if (!payload.notes.isEmpty()) {
            lines << QStringLiteral("Note:");
            for (const QString& note : payload.notes) {
                lines << QStringLiteral(" - %1").arg(note);
            }
        }
        if (roiOcrView_ != nullptr) {
            roiOcrView_->setPlainText(lines.join('\n'));
        }
        logView_->setPlainText(lines.join('\n'));
        statusLabel_->setText(QStringLiteral("OCR ROI completato."));
        if (ocrSizeButton_ != nullptr) {
            ocrSizeButton_->setEnabled(true);
        }
        if (ocrDotButton_ != nullptr) {
            ocrDotButton_->setEnabled(true);
        }
    });
    connect(&backend_, &BackendClient::roiOcrFailed, this, [this](const QString& errorText) {
        if (roiOcrView_ != nullptr) {
            roiOcrView_->setPlainText(errorText);
        }
        logView_->setPlainText(errorText);
        statusLabel_->setText(QStringLiteral("Errore OCR ROI."));
        if (ocrSizeButton_ != nullptr) {
            ocrSizeButton_->setEnabled(true);
        }
        if (ocrDotButton_ != nullptr) {
            ocrDotButton_->setEnabled(true);
        }
        QMessageBox::warning(this, QStringLiteral("Errore OCR ROI"), errorText);
    });
    connect(&backend_, &BackendClient::sectorPreviewStarted, this, [this](const QString&, const QString& branch) {
        statusLabel_->setText(QStringLiteral("Preview settore in corso (%1)...").arg(branch.toUpper()));
    });
    connect(&backend_, &BackendClient::sectorPreviewCompleted, this, [this]() {
        const auto& payload = backend_.lastSectorPayload();
        ImageViewer* target = payload.branch.compare(QStringLiteral("dot"), Qt::CaseInsensitive) == 0 ? dotPreviewViewer_ : sizePreviewViewer_;
        if (target != nullptr) {
            target->setImagePath(payload.unwrappedPath);
            target->resetView();
        }
        const QString sourceImagePath = currentImagePath();
        if (!sourceImagePath.isEmpty()) {
            auto& ann = annotationsByImage_[normalizePathKey(sourceImagePath)];
            if (payload.branch.compare(QStringLiteral("dot"), Qt::CaseInsensitive) == 0) {
                ann.dotPreviewPath = payload.unwrappedPath;
            } else {
                ann.sizePreviewPath = payload.unwrappedPath;
            }
        }
        QStringList lines;
        lines << QStringLiteral("Preview settore %1").arg(payload.branch.toUpper());
        lines << QStringLiteral("Angoli: %1 -> %2").arg(payload.startAngleDeg, 0, 'f', 2).arg(payload.endAngleDeg, 0, 'f', 2);
        lines << QStringLiteral("Output: %1").arg(payload.unwrappedPath);
        for (const auto& timing : payload.timings) {
            lines << QStringLiteral("%1: %2 ms").arg(timing.name).arg(QString::number(timing.ms, 'f', 3));
        }
        if (roiOcrView_ != nullptr) {
            roiOcrView_->setPlainText(lines.join('\n'));
        }
        statusLabel_->setText(QStringLiteral("Preview settore pronto."));
    });
    connect(&backend_, &BackendClient::sectorPreviewFailed, this, [this](const QString& errorText) {
        if (roiOcrView_ != nullptr) {
            roiOcrView_->setPlainText(errorText);
        }
        statusLabel_->setText(QStringLiteral("Errore preview settore."));
        QMessageBox::warning(this, QStringLiteral("Errore preview settore"), errorText);
    });
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("Tyre Reader Debug GUI"));
    resize(1680, 1000);

    auto* toolbar = addToolBar(QStringLiteral("Main"));
    toolbar->setMovable(false);
    toolbar->addAction(QStringLiteral("Apri cartella"), this, &MainWindow::openFolder);
    toolbar->addAction(QStringLiteral("Apri file"), this, &MainWindow::openFiles);
    toolbar->addAction(QStringLiteral("Apri annotazioni"), this, &MainWindow::openAnnotationsFile);
    toolbar->addAction(QStringLiteral("Salva annotazioni"), this, &MainWindow::saveAnnotationsFile);
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

    auto* annotationTab = new QWidget(this);
    auto* annotationLayout = new QVBoxLayout(annotationTab);
    annotationLayout->setContentsMargins(0, 0, 0, 0);
    annotationLayout->setSpacing(8);

    auto* annotationControls = new QHBoxLayout();
    auto* drawSizeButton = new QPushButton(QStringLiteral("Disegna SIZE"), this);
    auto* drawDotButton = new QPushButton(QStringLiteral("Disegna DOT"), this);
    auto* clearSizeButton = new QPushButton(QStringLiteral("Cancella SIZE"), this);
    auto* clearDotButton = new QPushButton(QStringLiteral("Cancella DOT"), this);
    ocrSizeButton_ = new QPushButton(QStringLiteral("OCR SIZE ROI"), this);
    ocrDotButton_ = new QPushButton(QStringLiteral("OCR DOT ROI"), this);
    auto* exportButton = new QPushButton(QStringLiteral("Esporta training set"), this);
    connect(drawSizeButton, &QPushButton::clicked, this, &MainWindow::startSizeAnnotation);
    connect(drawDotButton, &QPushButton::clicked, this, &MainWindow::startDotAnnotation);
    connect(clearSizeButton, &QPushButton::clicked, this, &MainWindow::clearSizeAnnotation);
    connect(clearDotButton, &QPushButton::clicked, this, &MainWindow::clearDotAnnotation);
    connect(applySizeSuggestionButton_, &QPushButton::clicked, this, &MainWindow::applySizeSuggestion);
    connect(applyDotSuggestionButton_, &QPushButton::clicked, this, &MainWindow::applyDotSuggestion);
    connect(ocrSizeButton_, &QPushButton::clicked, this, [this]() {
        const QString sourceImagePath = currentImagePath();
        const auto ann = annotationsByImage_.value(normalizePathKey(sourceImagePath));
        const QString annotationImagePath = ann.sizePreviewPath;
        if (sourceImagePath.isEmpty() || annotationImagePath.isEmpty() || backend_.isRoiBusy()) {
            return;
        }
        const QRect roi = previewCropRect(annotationImagePath, ann.sizeCropTopPercent, ann.sizeCropBottomPercent);
        if (roi.isNull() || roi.height() < 8) {
            QMessageBox::information(this, QStringLiteral("OCR SIZE ROI"), QStringLiteral("La fascia verticale SIZE e' troppo piccola."));
            return;
        }
        backend_.runRoiOcr(annotationImagePath, roi, QStringLiteral("size"));
    });
    connect(ocrDotButton_, &QPushButton::clicked, this, [this]() {
        const QString sourceImagePath = currentImagePath();
        const auto ann = annotationsByImage_.value(normalizePathKey(sourceImagePath));
        const QString annotationImagePath = ann.dotPreviewPath;
        if (sourceImagePath.isEmpty() || annotationImagePath.isEmpty() || backend_.isRoiBusy()) {
            return;
        }
        const QRect roi = previewCropRect(annotationImagePath, ann.dotCropTopPercent, ann.dotCropBottomPercent);
        if (roi.isNull() || roi.height() < 8) {
            QMessageBox::information(this, QStringLiteral("OCR DOT ROI"), QStringLiteral("La fascia verticale DOT e' troppo piccola."));
            return;
        }
        backend_.runRoiOcr(annotationImagePath, roi, QStringLiteral("dot"));
    });
    connect(exportButton, &QPushButton::clicked, this, &MainWindow::exportAnnotations);
    annotationControls->addWidget(drawSizeButton);
    annotationControls->addWidget(drawDotButton);
    annotationControls->addWidget(clearSizeButton);
    annotationControls->addWidget(clearDotButton);
    annotationControls->addWidget(ocrSizeButton_);
    annotationControls->addWidget(ocrDotButton_);
    annotationControls->addWidget(exportButton);
    annotationControls->addStretch(1);
    annotationLayout->addLayout(annotationControls);

    auto* metadataLayout = new QHBoxLayout();
    sizeTextEdit_ = new QLineEdit(this);
    sizeTextEdit_->setPlaceholderText(QStringLiteral("Testo atteso SIZE, es. 195/55 R15"));
    dotTextEdit_ = new QLineEdit(this);
    dotTextEdit_->setPlaceholderText(QStringLiteral("Testo atteso DOT o DOT parziale"));
    dot4TextEdit_ = new QLineEdit(this);
    dot4TextEdit_->setPlaceholderText(QStringLiteral("DOT WWYY atteso"));
    sizeVisibleCheck_ = new QCheckBox(QStringLiteral("SIZE visibile"), this);
    sizeVisibleCheck_->setChecked(true);
    dotVisibleCheck_ = new QCheckBox(QStringLiteral("DOT visibile"), this);
    dotVisibleCheck_->setChecked(true);
    dotDateVisibleCheck_ = new QCheckBox(QStringLiteral("DOT data visibile"), this);
    metadataLayout->addWidget(sizeTextEdit_, 2);
    metadataLayout->addWidget(dotTextEdit_, 2);
    metadataLayout->addWidget(dot4TextEdit_, 1);
    metadataLayout->addWidget(sizeVisibleCheck_);
    metadataLayout->addWidget(dotVisibleCheck_);
    metadataLayout->addWidget(dotDateVisibleCheck_);
    annotationLayout->addLayout(metadataLayout);

    auto* suggestionLayout = new QVBoxLayout();
    auto* suggestionRow1 = new QHBoxLayout();
    suggestedSizeEdit_ = new QLineEdit(this);
    suggestedSizeEdit_->setReadOnly(true);
    suggestedSizeEdit_->setPlaceholderText(QStringLiteral("Suggerimento SIZE"));
    suggestedSizeMetaEdit_ = new QLineEdit(this);
    suggestedSizeMetaEdit_->setReadOnly(true);
    suggestedSizeMetaEdit_->setPlaceholderText(QStringLiteral("Confidenza / posizione SIZE"));
    applySizeSuggestionButton_ = new QPushButton(QStringLiteral("Usa SIZE suggerita"), this);
    suggestionRow1->addWidget(suggestedSizeEdit_, 3);
    suggestionRow1->addWidget(suggestedSizeMetaEdit_, 2);
    suggestionRow1->addWidget(applySizeSuggestionButton_);
    suggestionLayout->addLayout(suggestionRow1);

    auto* suggestionRow2 = new QHBoxLayout();
    suggestedDotEdit_ = new QLineEdit(this);
    suggestedDotEdit_->setReadOnly(true);
    suggestedDotEdit_->setPlaceholderText(QStringLiteral("Suggerimento DOT"));
    suggestedDot4Edit_ = new QLineEdit(this);
    suggestedDot4Edit_->setReadOnly(true);
    suggestedDot4Edit_->setPlaceholderText(QStringLiteral("Suggerimento DOT4"));
    suggestedDotMetaEdit_ = new QLineEdit(this);
    suggestedDotMetaEdit_->setReadOnly(true);
    suggestedDotMetaEdit_->setPlaceholderText(QStringLiteral("Confidenza / posizione DOT"));
    applyDotSuggestionButton_ = new QPushButton(QStringLiteral("Usa DOT suggerito"), this);
    suggestionRow2->addWidget(suggestedDotEdit_, 3);
    suggestionRow2->addWidget(suggestedDot4Edit_, 1);
    suggestionRow2->addWidget(suggestedDotMetaEdit_, 2);
    suggestionRow2->addWidget(applyDotSuggestionButton_);
    suggestionLayout->addLayout(suggestionRow2);

    suggestionNotesView_ = new QPlainTextEdit(this);
    suggestionNotesView_->setReadOnly(true);
    suggestionNotesView_->setPlaceholderText(QStringLiteral("Note del suggerimento"));
    suggestionNotesView_->setMaximumBlockCount(50);
    suggestionNotesView_->setMinimumHeight(60);
    suggestionLayout->addWidget(suggestionNotesView_);
    annotationLayout->addLayout(suggestionLayout);

    auto makeAngleSpin = [this]() {
        auto* spin = new QDoubleSpinBox(this);
        spin->setRange(0.0, 360.0);
        spin->setDecimals(2);
        spin->setSingleStep(1.0);
        return spin;
    };
    auto makeWheelSpin = [this]() {
        auto* spin = new QDoubleSpinBox(this);
        spin->setRange(-5000.0, 5000.0);
        spin->setDecimals(2);
        spin->setSingleStep(1.0);
        return spin;
    };

    auto* sectorEditLayout = new QHBoxLayout();
    sizeSectorStartSpin_ = makeAngleSpin();
    sizeSectorEndSpin_ = makeAngleSpin();
    dotSectorStartSpin_ = makeAngleSpin();
    dotSectorEndSpin_ = makeAngleSpin();
    auto* applySizeSectorButton = new QPushButton(QStringLiteral("Aggiorna SIZE"), this);
    auto* applyDotSectorButton = new QPushButton(QStringLiteral("Aggiorna DOT"), this);
    connect(sizeSectorStartSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &MainWindow::sizeSectorStartChanged);
    connect(sizeSectorEndSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &MainWindow::sizeSectorEndChanged);
    connect(dotSectorStartSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &MainWindow::dotSectorStartChanged);
    connect(dotSectorEndSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &MainWindow::dotSectorEndChanged);
    connect(applySizeSectorButton, &QPushButton::clicked, this, &MainWindow::applySizeSectorEdits);
    connect(applyDotSectorButton, &QPushButton::clicked, this, &MainWindow::applyDotSectorEdits);
    sectorEditLayout->addWidget(new QLabel(QStringLiteral("SIZE angoli"), this));
    sectorEditLayout->addWidget(sizeSectorStartSpin_);
    sectorEditLayout->addWidget(sizeSectorEndSpin_);
    sectorEditLayout->addWidget(applySizeSectorButton);
    sectorEditLayout->addSpacing(12);
    sectorEditLayout->addWidget(new QLabel(QStringLiteral("DOT angoli"), this));
    sectorEditLayout->addWidget(dotSectorStartSpin_);
    sectorEditLayout->addWidget(dotSectorEndSpin_);
    sectorEditLayout->addWidget(applyDotSectorButton);
    annotationLayout->addLayout(sectorEditLayout);

    auto* wheelEditLayout = new QHBoxLayout();
    wheelCenterXSpin_ = makeWheelSpin();
    wheelCenterYSpin_ = makeWheelSpin();
    wheelInnerRadiusSpin_ = makeWheelSpin();
    wheelOuterRadiusSpin_ = makeWheelSpin();
    auto* applyWheelButton = new QPushButton(QStringLiteral("Applica corona"), this);
    connect(applyWheelButton, &QPushButton::clicked, this, &MainWindow::applyWheelOverride);
    wheelEditLayout->addWidget(new QLabel(QStringLiteral("Cx"), this));
    wheelEditLayout->addWidget(wheelCenterXSpin_);
    wheelEditLayout->addWidget(new QLabel(QStringLiteral("Cy"), this));
    wheelEditLayout->addWidget(wheelCenterYSpin_);
    wheelEditLayout->addWidget(new QLabel(QStringLiteral("Rin"), this));
    wheelEditLayout->addWidget(wheelInnerRadiusSpin_);
    wheelEditLayout->addWidget(new QLabel(QStringLiteral("Rout"), this));
    wheelEditLayout->addWidget(wheelOuterRadiusSpin_);
    wheelEditLayout->addWidget(applyWheelButton);
    annotationLayout->addLayout(wheelEditLayout);

    auto* cropLayout = new QHBoxLayout();
    sizeCropLabel_ = new QLabel(QStringLiteral("SIZE Y: 0-100%"), this);
    sizeCropTopSlider_ = new QSlider(Qt::Horizontal, this);
    sizeCropBottomSlider_ = new QSlider(Qt::Horizontal, this);
    dotCropLabel_ = new QLabel(QStringLiteral("DOT Y: 0-100%"), this);
    dotCropTopSlider_ = new QSlider(Qt::Horizontal, this);
    dotCropBottomSlider_ = new QSlider(Qt::Horizontal, this);
    for (QSlider* slider : {sizeCropTopSlider_, sizeCropBottomSlider_, dotCropTopSlider_, dotCropBottomSlider_}) {
        slider->setRange(0, 100);
    }
    sizeCropTopSlider_->setValue(0);
    sizeCropBottomSlider_->setValue(100);
    dotCropTopSlider_->setValue(0);
    dotCropBottomSlider_->setValue(100);
    connect(sizeCropTopSlider_, &QSlider::valueChanged, this, &MainWindow::sizeCropTopChanged);
    connect(sizeCropBottomSlider_, &QSlider::valueChanged, this, &MainWindow::sizeCropBottomChanged);
    connect(dotCropTopSlider_, &QSlider::valueChanged, this, &MainWindow::dotCropTopChanged);
    connect(dotCropBottomSlider_, &QSlider::valueChanged, this, &MainWindow::dotCropBottomChanged);
    cropLayout->addWidget(sizeCropLabel_);
    cropLayout->addWidget(sizeCropTopSlider_);
    cropLayout->addWidget(sizeCropBottomSlider_);
    cropLayout->addSpacing(16);
    cropLayout->addWidget(dotCropLabel_);
    cropLayout->addWidget(dotCropTopSlider_);
    cropLayout->addWidget(dotCropBottomSlider_);
    annotationLayout->addLayout(cropLayout);

    auto syncMetadata = [this]() {
        const QString imagePath = currentImagePath();
        if (imagePath.isEmpty()) {
            return;
        }
        ImageAnnotations& ann = annotationsByImage_[normalizePathKey(imagePath)];
        ann.sizeText = sizeTextEdit_->text();
        ann.dotText = dotTextEdit_->text();
        ann.dot4Text = dot4TextEdit_->text();
        ann.sizeVisible = sizeVisibleCheck_->isChecked();
        ann.dotVisible = dotVisibleCheck_->isChecked();
        ann.dotDateVisible = dotDateVisibleCheck_->isChecked();
        autoSaveAnnotations();
    };
    connect(sizeTextEdit_, &QLineEdit::textChanged, this, [syncMetadata]() { syncMetadata(); });
    connect(dotTextEdit_, &QLineEdit::textChanged, this, [syncMetadata]() { syncMetadata(); });
    connect(dot4TextEdit_, &QLineEdit::textChanged, this, [syncMetadata]() { syncMetadata(); });
    connect(sizeVisibleCheck_, &QCheckBox::toggled, this, [syncMetadata](bool) { syncMetadata(); });
    connect(dotVisibleCheck_, &QCheckBox::toggled, this, [syncMetadata](bool) { syncMetadata(); });
    connect(dotDateVisibleCheck_, &QCheckBox::toggled, this, [syncMetadata](bool) { syncMetadata(); });

    auto* previewSplit = new QSplitter(Qt::Horizontal, this);
    annotationViewer_ = new ImageViewer(this);
    sizePreviewViewer_ = new ImageViewer(this);
    dotPreviewViewer_ = new ImageViewer(this);
    previewSplit->addWidget(annotationViewer_);
    previewSplit->addWidget(sizePreviewViewer_);
    previewSplit->addWidget(dotPreviewViewer_);
    previewSplit->setStretchFactor(0, 2);
    previewSplit->setStretchFactor(1, 1);
    previewSplit->setStretchFactor(2, 1);
    annotationLayout->addWidget(previewSplit, 1);

    roiOcrView_ = new QPlainTextEdit(this);
    roiOcrView_->setReadOnly(true);
    roiOcrView_->setPlaceholderText(QStringLiteral("Qui compariranno testo e tempi dell'OCR sulla ROI selezionata."));
    roiOcrView_->setMaximumBlockCount(200);
    roiOcrView_->setMinimumHeight(130);
    annotationLayout->addWidget(roiOcrView_);

    tabWidget->addTab(annotationTab, QStringLiteral("Annotazione"));

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
    if (applySizeSuggestionButton_ != nullptr) {
        applySizeSuggestionButton_->setEnabled(false);
    }
    if (applyDotSuggestionButton_ != nullptr) {
        applyDotSuggestionButton_->setEnabled(false);
    }
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
    annotationsFilePath_ = QDir(folder).filePath(QStringLiteral(".tyre_reader_annotations.json"));
    annotationsByImage_.clear();
    loadAnnotationsFromFile(annotationsFilePath_);
    suggestionsFilePath_ = QDir(folder).filePath(QStringLiteral("training_set_suggestions/assistant_suggestions.csv"));
    suggestionsByImage_.clear();
    loadSuggestionsFromCsv(suggestionsFilePath_);
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
        annotationsFilePath_ = defaultAnnotationsPathForCurrentSelection();
        annotationsByImage_.clear();
        if (!files.isEmpty()) {
            const QFileInfo firstInfo(files.first());
            annotationsFilePath_ = QDir(firstInfo.absolutePath()).filePath(QStringLiteral(".tyre_reader_annotations.json"));
            loadAnnotationsFromFile(annotationsFilePath_);
            suggestionsFilePath_ = QDir(firstInfo.absolutePath()).filePath(QStringLiteral("training_set_suggestions/assistant_suggestions.csv"));
            suggestionsByImage_.clear();
            loadSuggestionsFromCsv(suggestionsFilePath_);
        }
        setImageList(files);
    }
}

void MainWindow::setImageList(const QStringList& files) {
    imagePaths_ = files;
    if (!suggestionsFilePath_.isEmpty()) {
        suggestionsByImage_.clear();
        loadSuggestionsFromCsv(suggestionsFilePath_.isEmpty() ? defaultSuggestionsPathForCurrentSelection() : suggestionsFilePath_);
    }
    galleryList_->clear();
    stepList_->clear();
    imageViewer_->setImagePath(QString());
    imageViewer_->resetView();
    annotationViewer_->setImagePath(QString());
    annotationViewer_->resetView();
    if (roiOcrView_ != nullptr) {
        roiOcrView_->clear();
    }
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
    refreshAnnotationViewer();
    const auto ann = annotationsByImage_.value(normalizePathKey(imagePath));
    syncControlsFromAnnotations(ann, payloadsBySourceImage_.value(normalizePathKey(imagePath)));
    refreshSuggestionPanel();
    resultsTable_->setRowCount(0);
    timingsTable_->setRowCount(0);
    logView_->setPlainText(QStringLiteral("Immagine selezionata. Premi 'Start elaborazione' per avviare il backend."));
    if (roiOcrView_ != nullptr) {
        const auto annForUi = annotationsByImage_.value(normalizePathKey(imagePath));
        if (annForUi.sizePreviewPath.isEmpty() && annForUi.dotPreviewPath.isEmpty()) {
            roiOcrView_->setPlainText(QStringLiteral("Esegui prima l'elaborazione per ottenere la geometria ruota, poi disegna un settore SIZE o DOT sulla corona."));
        } else {
            roiOcrView_->setPlainText(QStringLiteral("Disegna un settore sulla corona. Le preview rettangolari SIZE e DOT compariranno nei riquadri a destra."));
        }
    }
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
    const auto& payload = backend_.lastPayload();
    payloadsBySourceImage_[normalizePathKey(payload.inputPath)] = payload;
    auto& ann = annotationsByImage_[normalizePathKey(payload.inputPath)];
    if (!ann.wheelOverrideSet && payload.wheelFound) {
        ann.wheelCenterX = payload.wheelCenterX;
        ann.wheelCenterY = payload.wheelCenterY;
        ann.wheelInnerRadius = payload.wheelInnerRadius;
        ann.wheelOuterRadius = payload.wheelOuterRadius;
    }
    updateFromPayload(backend_.lastPayload());
    refreshAnnotationViewer();
    autoSaveAnnotations();
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

    const QString sourceImagePath = currentImagePath();
    if (!sourceImagePath.isEmpty() && normalizePathKey(sourceImagePath) == normalizePathKey(payload.inputPath)) {
        ImageAnnotations& ann = annotationsByImage_[normalizePathKey(sourceImagePath)];
        const QString defaultUnwrap = resolveDefaultUnwrapImagePath(payload);
        if (!defaultUnwrap.isEmpty()) {
            if (ann.sizePreviewPath.isEmpty()) {
                ann.sizePreviewPath = defaultUnwrap;
            }
            if (ann.dotPreviewPath.isEmpty()) {
                ann.dotPreviewPath = defaultUnwrap;
            }
        }
    }
    autoSaveAnnotations();
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

void MainWindow::refreshAnnotationViewer() {
    const QString sourceImagePath = currentImagePath();
    const auto payload = payloadsBySourceImage_.value(normalizePathKey(sourceImagePath));
    const auto ann = annotationsByImage_.value(normalizePathKey(sourceImagePath));
    const QString annotationImagePath = resolveWheelAnnotationImagePath(payload).isEmpty()
        ? sourceImagePath
        : resolveWheelAnnotationImagePath(payload);
    annotationViewer_->setImagePath(annotationImagePath);
    annotationViewer_->resetView();
    const bool wheelFound = ann.wheelOverrideSet ? true : payload.wheelFound;
    const QPointF wheelCenter = ann.wheelOverrideSet ? QPointF(ann.wheelCenterX, ann.wheelCenterY)
                                                     : QPointF(payload.wheelCenterX, payload.wheelCenterY);
    const double innerRadius = ann.wheelOverrideSet ? ann.wheelInnerRadius : payload.wheelInnerRadius;
    const double outerRadius = ann.wheelOverrideSet ? ann.wheelOuterRadius : payload.wheelOuterRadius;
    annotationViewer_->setWheelGeometry(wheelFound, wheelCenter, innerRadius, outerRadius);

    QVector<ImageViewer::AnnotationBox> boxes;
    QVector<ImageViewer::SectorAnnotation> sectors;
    const auto it = annotationsByImage_.constFind(normalizePathKey(sourceImagePath));
    if (it != annotationsByImage_.constEnd()) {
        if (it->sizeSectorSet) {
            sectors.push_back({QStringLiteral("SIZE"), Qt::red, it->sizeStartAngleDeg, it->sizeEndAngleDeg});
        }
        if (it->dotSectorSet) {
            sectors.push_back({QStringLiteral("DOT"), Qt::cyan, it->dotStartAngleDeg, it->dotEndAngleDeg});
        }
        if (sizePreviewViewer_ != nullptr) {
            const QString sizePath = !it->sizePreviewPath.isEmpty() ? it->sizePreviewPath : resolveDefaultUnwrapImagePath(payload);
            sizePreviewViewer_->setImagePath(sizePath);
            sizePreviewViewer_->resetView();
        }
        if (dotPreviewViewer_ != nullptr) {
            const QString dotPath = !it->dotPreviewPath.isEmpty() ? it->dotPreviewPath : resolveDefaultUnwrapImagePath(payload);
            dotPreviewViewer_->setImagePath(dotPath);
            dotPreviewViewer_->resetView();
        }
    } else {
        if (sizePreviewViewer_ != nullptr) {
            sizePreviewViewer_->setImagePath(resolveDefaultUnwrapImagePath(payload));
        }
        if (dotPreviewViewer_ != nullptr) {
            dotPreviewViewer_->setImagePath(resolveDefaultUnwrapImagePath(payload));
        }
    }
    annotationViewer_->setAnnotationBoxes(boxes);
    annotationViewer_->setSectorAnnotations(sectors);
    annotationViewer_->setInteractionMode(ImageViewer::InteractionMode::View);
    syncControlsFromAnnotations(ann, payload);
    refreshPreviewCropBoxes();
    refreshSuggestionPanel();
}

void MainWindow::startSizeAnnotation() {
    annotationViewer_->setInteractionMode(ImageViewer::InteractionMode::DrawSizeSector);
    statusLabel_->setText(QStringLiteral("Traccia un settore SIZE sulla corona della ruota."));
}

void MainWindow::startDotAnnotation() {
    annotationViewer_->setInteractionMode(ImageViewer::InteractionMode::DrawDotSector);
    statusLabel_->setText(QStringLiteral("Traccia un settore DOT sulla corona della ruota."));
}

void MainWindow::clearSizeAnnotation() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    annotationsByImage_[normalizePathKey(imagePath)].sizeSectorSet = false;
    annotationsByImage_[normalizePathKey(imagePath)].sizePreviewPath.clear();
    annotationsByImage_[normalizePathKey(imagePath)].sizeCropTopPercent = 0;
    annotationsByImage_[normalizePathKey(imagePath)].sizeCropBottomPercent = 100;
    refreshAnnotationViewer();
    autoSaveAnnotations();
}

void MainWindow::clearDotAnnotation() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    annotationsByImage_[normalizePathKey(imagePath)].dotSectorSet = false;
    annotationsByImage_[normalizePathKey(imagePath)].dotPreviewPath.clear();
    annotationsByImage_[normalizePathKey(imagePath)].dotCropTopPercent = 0;
    annotationsByImage_[normalizePathKey(imagePath)].dotCropBottomPercent = 100;
    refreshAnnotationViewer();
    autoSaveAnnotations();
}

void MainWindow::exportAnnotations() {
    const QString exportDirPath = QFileDialog::getExistingDirectory(
        this,
        QStringLiteral("Esporta training set"),
        QStringLiteral("training_set_export"));
    if (exportDirPath.isEmpty()) {
        return;
    }

    QDir exportDir(exportDirPath);
    if (!exportDir.exists() && !QDir().mkpath(exportDirPath)) {
        QMessageBox::warning(this, QStringLiteral("Export annotazioni"), QStringLiteral("Impossibile creare la directory di output."));
        return;
    }

    const QString sizeDirPath = exportDir.filePath(QStringLiteral("size_crops"));
    const QString dotDirPath = exportDir.filePath(QStringLiteral("dot_crops"));
    QDir().mkpath(sizeDirPath);
    QDir().mkpath(dotDirPath);

    QFile manifestFile(exportDir.filePath(QStringLiteral("annotations.csv")));
    QFile sizeManifestFile(exportDir.filePath(QStringLiteral("size_crops.csv")));
    QFile dotManifestFile(exportDir.filePath(QStringLiteral("dot_crops.csv")));
    if (!manifestFile.open(QIODevice::WriteOnly | QIODevice::Text) ||
        !sizeManifestFile.open(QIODevice::WriteOnly | QIODevice::Text) ||
        !dotManifestFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, QStringLiteral("Export annotazioni"), QStringLiteral("Impossibile aprire i file manifest del training set."));
        return;
    }

    QTextStream manifest(&manifestFile);
    QTextStream sizeManifest(&sizeManifestFile);
    QTextStream dotManifest(&dotManifestFile);
    manifest << "image,size_start_deg,size_end_deg,size_preview,size_text,size_visible,dot_start_deg,dot_end_deg,dot_preview,dot_text,dot_visible,dot_date_visible,dot4_text\n";
    sizeManifest << "image,annotation_image,crop,size_start_deg,size_end_deg,size_text,size_visible\n";
    dotManifest << "image,annotation_image,crop,dot_start_deg,dot_end_deg,dot_text,dot_visible,dot_date_visible,dot4_text\n";

    int sizeCropCount = 0;
    int dotCropCount = 0;

    for (auto it = annotationsByImage_.cbegin(); it != annotationsByImage_.cend(); ++it) {
        QString escapedPath = it.key();
        escapedPath.replace("\"", "\"\"");
        const auto payloadIt = payloadsBySourceImage_.constFind(normalizePathKey(it.key()));
        QString annotationImagePath = (payloadIt != payloadsBySourceImage_.constEnd() && !payloadIt->wheelOverlayPath.isEmpty())
            ? payloadIt->wheelOverlayPath
            : it.key();
        QString escapedAnnotationImagePath = annotationImagePath;
        escapedAnnotationImagePath.replace("\"", "\"\"");
        QString escapedSizeText = it->sizeText;
        QString escapedDotText = it->dotText;
        QString escapedDot4Text = it->dot4Text;
        QString escapedSizePreview = it->sizePreviewPath;
        QString escapedDotPreview = it->dotPreviewPath;
        escapedSizeText.replace("\"", "\"\"");
        escapedDotText.replace("\"", "\"\"");
        escapedDot4Text.replace("\"", "\"\"");
        escapedSizePreview.replace("\"", "\"\"");
        escapedDotPreview.replace("\"", "\"\"");

        manifest << "\"" << escapedPath << "\","
                 << it->sizeStartAngleDeg << ","
                 << it->sizeEndAngleDeg << ","
                 << "\"" << escapedSizePreview << "\","
                 << "\"" << escapedSizeText << "\","
                 << (it->sizeVisible ? "true" : "false") << ","
                 << it->dotStartAngleDeg << ","
                 << it->dotEndAngleDeg << ","
                 << "\"" << escapedDotPreview << "\","
                 << "\"" << escapedDotText << "\","
                 << (it->dotVisible ? "true" : "false") << ","
                 << (it->dotDateVisible ? "true" : "false") << ","
                 << "\"" << escapedDot4Text << "\"\n";

        if (!it->sizePreviewPath.isEmpty() && QFileInfo::exists(it->sizePreviewPath)) {
            const QString cropName = QStringLiteral("%1_size.png").arg(QFileInfo(it.key()).completeBaseName());
            const QString cropPath = QDir(sizeDirPath).filePath(cropName);
            QFile::remove(cropPath);
            QFile::copy(it->sizePreviewPath, cropPath);
            QString escapedCropPath = cropPath;
            escapedCropPath.replace("\"", "\"\"");
            sizeManifest << "\"" << escapedPath << "\","
                         << "\"" << escapedAnnotationImagePath << "\","
                         << "\"" << escapedCropPath << "\","
                         << it->sizeStartAngleDeg << ","
                         << it->sizeEndAngleDeg << ","
                         << "\"" << escapedSizeText << "\","
                         << (it->sizeVisible ? "true" : "false") << "\n";
            ++sizeCropCount;
        }

        if (!it->dotPreviewPath.isEmpty() && QFileInfo::exists(it->dotPreviewPath)) {
            const QString cropName = QStringLiteral("%1_dot.png").arg(QFileInfo(it.key()).completeBaseName());
            const QString cropPath = QDir(dotDirPath).filePath(cropName);
            QFile::remove(cropPath);
            QFile::copy(it->dotPreviewPath, cropPath);
            QString escapedCropPath = cropPath;
            escapedCropPath.replace("\"", "\"\"");
            dotManifest << "\"" << escapedPath << "\","
                        << "\"" << escapedAnnotationImagePath << "\","
                        << "\"" << escapedCropPath << "\","
                        << it->dotStartAngleDeg << ","
                        << it->dotEndAngleDeg << ","
                        << "\"" << escapedDotText << "\","
                        << (it->dotVisible ? "true" : "false") << ","
                        << (it->dotDateVisible ? "true" : "false") << ","
                        << "\"" << escapedDot4Text << "\"\n";
            ++dotCropCount;
        }
    }
    statusLabel_->setText(QStringLiteral("Training set esportato in %1").arg(exportDirPath));
    saveAnnotationsToFile(exportDir.filePath(QStringLiteral("annotations.json")));
    if (roiOcrView_ != nullptr) {
        roiOcrView_->setPlainText(
            QStringLiteral("Export completato.\nDirectory: %1\nCrop SIZE: %2\nCrop DOT: %3")
                .arg(exportDirPath)
                .arg(sizeCropCount)
                .arg(dotCropCount));
    }
}

void MainWindow::handleAnnotationDrawn(const QString& label, const QRect& imageRect) {
    Q_UNUSED(label);
    Q_UNUSED(imageRect);
}

void MainWindow::handleSectorDrawn(const QString& label, double startAngleDeg, double endAngleDeg) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }

    ImageAnnotations& ann = annotationsByImage_[normalizePathKey(imagePath)];
    const bool isDot = label == QStringLiteral("DOT");
    if (isDot) {
        ann.dotStartAngleDeg = startAngleDeg;
        ann.dotEndAngleDeg = endAngleDeg;
        ann.dotSectorSet = true;
        ann.dotPreviewPath.clear();
        ann.dotCropTopPercent = 0;
        ann.dotCropBottomPercent = 100;
    } else {
        ann.sizeStartAngleDeg = startAngleDeg;
        ann.sizeEndAngleDeg = endAngleDeg;
        ann.sizeSectorSet = true;
        ann.sizePreviewPath.clear();
        ann.sizeCropTopPercent = 0;
        ann.sizeCropBottomPercent = 100;
    }

    annotationViewer_->setInteractionMode(ImageViewer::InteractionMode::View);
    refreshAnnotationViewer();
    backend_.runSectorPreview(imagePath,
                              isDot ? QStringLiteral("dot") : QStringLiteral("size"),
                              startAngleDeg,
                              endAngleDeg,
                              ann.wheelOverrideSet,
                              ann.wheelCenterX,
                              ann.wheelCenterY,
                              ann.wheelInnerRadius,
                              ann.wheelOuterRadius);
    autoSaveAnnotations();
    statusLabel_->setText(QStringLiteral("Settore %1 inviato al backend per unwrap locale.").arg(label));
}

void MainWindow::sizeCropTopChanged(int value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.sizeCropTopPercent = std::min(value, ann.sizeCropBottomPercent - 1);
    if (sizeCropTopSlider_->value() != ann.sizeCropTopPercent) {
        const QSignalBlocker blocker(sizeCropTopSlider_);
        sizeCropTopSlider_->setValue(ann.sizeCropTopPercent);
    }
    refreshPreviewCropBoxes();
    autoSaveAnnotations();
}

void MainWindow::sizeCropBottomChanged(int value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.sizeCropBottomPercent = std::max(value, ann.sizeCropTopPercent + 1);
    if (sizeCropBottomSlider_->value() != ann.sizeCropBottomPercent) {
        const QSignalBlocker blocker(sizeCropBottomSlider_);
        sizeCropBottomSlider_->setValue(ann.sizeCropBottomPercent);
    }
    refreshPreviewCropBoxes();
    autoSaveAnnotations();
}

void MainWindow::dotCropTopChanged(int value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.dotCropTopPercent = std::min(value, ann.dotCropBottomPercent - 1);
    if (dotCropTopSlider_->value() != ann.dotCropTopPercent) {
        const QSignalBlocker blocker(dotCropTopSlider_);
        dotCropTopSlider_->setValue(ann.dotCropTopPercent);
    }
    refreshPreviewCropBoxes();
    autoSaveAnnotations();
}

void MainWindow::dotCropBottomChanged(int value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.dotCropBottomPercent = std::max(value, ann.dotCropTopPercent + 1);
    if (dotCropBottomSlider_->value() != ann.dotCropBottomPercent) {
        const QSignalBlocker blocker(dotCropBottomSlider_);
        dotCropBottomSlider_->setValue(ann.dotCropBottomPercent);
    }
    refreshPreviewCropBoxes();
    autoSaveAnnotations();
}

void MainWindow::sizeSectorStartChanged(double value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    annotationsByImage_[normalizePathKey(imagePath)].sizeStartAngleDeg = value;
    refreshAnnotationViewer();
}

void MainWindow::sizeSectorEndChanged(double value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    annotationsByImage_[normalizePathKey(imagePath)].sizeEndAngleDeg = value;
    refreshAnnotationViewer();
}

void MainWindow::dotSectorStartChanged(double value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    annotationsByImage_[normalizePathKey(imagePath)].dotStartAngleDeg = value;
    refreshAnnotationViewer();
}

void MainWindow::dotSectorEndChanged(double value) {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    annotationsByImage_[normalizePathKey(imagePath)].dotEndAngleDeg = value;
    refreshAnnotationViewer();
}

void MainWindow::applySizeSectorEdits() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.sizeSectorSet = true;
    ann.sizePreviewPath.clear();
    backend_.runSectorPreview(imagePath,
                              QStringLiteral("size"),
                              ann.sizeStartAngleDeg,
                              ann.sizeEndAngleDeg,
                              ann.wheelOverrideSet,
                              ann.wheelCenterX,
                              ann.wheelCenterY,
                              ann.wheelInnerRadius,
                              ann.wheelOuterRadius);
    autoSaveAnnotations();
}

void MainWindow::applyDotSectorEdits() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.dotSectorSet = true;
    ann.dotPreviewPath.clear();
    backend_.runSectorPreview(imagePath,
                              QStringLiteral("dot"),
                              ann.dotStartAngleDeg,
                              ann.dotEndAngleDeg,
                              ann.wheelOverrideSet,
                              ann.wheelCenterX,
                              ann.wheelCenterY,
                              ann.wheelInnerRadius,
                              ann.wheelOuterRadius);
    autoSaveAnnotations();
}

void MainWindow::applyWheelOverride() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) return;
    auto& ann = annotationsByImage_[normalizePathKey(imagePath)];
    ann.wheelOverrideSet = true;
    ann.wheelCenterX = wheelCenterXSpin_->value();
    ann.wheelCenterY = wheelCenterYSpin_->value();
    ann.wheelInnerRadius = wheelInnerRadiusSpin_->value();
    ann.wheelOuterRadius = wheelOuterRadiusSpin_->value();
    refreshAnnotationViewer();
    autoSaveAnnotations();
}

void MainWindow::openAnnotationsFile() {
    const QString filePath = QFileDialog::getOpenFileName(this, QStringLiteral("Apri annotazioni"), QString(), QStringLiteral("JSON (*.json)"));
    if (filePath.isEmpty()) return;
    loadAnnotationsFromFile(filePath);
    annotationsFilePath_ = filePath;
    refreshAnnotationViewer();
}

void MainWindow::saveAnnotationsFile() {
    QString filePath = annotationsFilePath_;
    if (filePath.isEmpty()) {
        filePath = QFileDialog::getSaveFileName(this, QStringLiteral("Salva annotazioni"), defaultAnnotationsPathForCurrentSelection(), QStringLiteral("JSON (*.json)"));
        if (filePath.isEmpty()) return;
        annotationsFilePath_ = filePath;
    }
    saveAnnotationsToFile(filePath);
    statusLabel_->setText(QStringLiteral("Annotazioni salvate in %1").arg(filePath));
}

void MainWindow::applySizeSuggestion() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    const auto it = suggestionsByImage_.constFind(normalizePathKey(imagePath));
    if (it == suggestionsByImage_.constEnd()) {
        return;
    }
    const QString suggestion = !it->sizeFullGuess.isEmpty() ? it->sizeFullGuess : it->sizeBaseGuess;
    if (suggestion.isEmpty()) {
        return;
    }
    sizeTextEdit_->setText(suggestion);
    sizeVisibleCheck_->setChecked(true);
    statusLabel_->setText(QStringLiteral("Suggerimento SIZE copiato nei campi modificabili."));
}

void MainWindow::applyDotSuggestion() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    const auto it = suggestionsByImage_.constFind(normalizePathKey(imagePath));
    if (it == suggestionsByImage_.constEnd()) {
        return;
    }
    if (!it->dotTextGuess.isEmpty()) {
        dotTextEdit_->setText(it->dotTextGuess);
        dotVisibleCheck_->setChecked(true);
    }
    if (!it->dot4Guess.isEmpty()) {
        dot4TextEdit_->setText(it->dot4Guess);
        dotDateVisibleCheck_->setChecked(true);
    }
    statusLabel_->setText(QStringLiteral("Suggerimento DOT copiato nei campi modificabili."));
}

QString MainWindow::currentAnnotationImagePath() const {
    return resolveAnnotationImagePath(currentImagePath());
}

void MainWindow::refreshPreviewCropBoxes() {
    const QString imagePath = currentImagePath();
    if (imagePath.isEmpty()) {
        return;
    }
    const auto ann = annotationsByImage_.value(normalizePathKey(imagePath));
    if (sizeCropLabel_ != nullptr) {
        sizeCropLabel_->setText(QStringLiteral("SIZE Y: %1-%2%").arg(ann.sizeCropTopPercent).arg(ann.sizeCropBottomPercent));
    }
    if (dotCropLabel_ != nullptr) {
        dotCropLabel_->setText(QStringLiteral("DOT Y: %1-%2%").arg(ann.dotCropTopPercent).arg(ann.dotCropBottomPercent));
    }
    if (sizePreviewViewer_ != nullptr) {
        QVector<ImageViewer::AnnotationBox> boxes;
        const QRect rect = previewCropRect(ann.sizePreviewPath, ann.sizeCropTopPercent, ann.sizeCropBottomPercent);
        if (!rect.isNull()) {
            boxes.push_back({QStringLiteral("SIZE OCR"), rect, Qt::yellow});
        }
        sizePreviewViewer_->setAnnotationBoxes(boxes);
    }
    if (dotPreviewViewer_ != nullptr) {
        QVector<ImageViewer::AnnotationBox> boxes;
        const QRect rect = previewCropRect(ann.dotPreviewPath, ann.dotCropTopPercent, ann.dotCropBottomPercent);
        if (!rect.isNull()) {
            boxes.push_back({QStringLiteral("DOT OCR"), rect, Qt::yellow});
        }
        dotPreviewViewer_->setAnnotationBoxes(boxes);
    }
}

QRect MainWindow::previewCropRect(const QString& previewPath, int topPercent, int bottomPercent) const {
    if (previewPath.isEmpty()) {
        return {};
    }
    QImage image(previewPath);
    if (image.isNull()) {
        return {};
    }
    const int h = image.height();
    const int w = image.width();
    const int y0 = std::clamp(static_cast<int>(std::round(h * (topPercent / 100.0))), 0, std::max(0, h - 1));
    const int y1 = std::clamp(static_cast<int>(std::round(h * (bottomPercent / 100.0))), y0 + 1, h);
    return QRect(0, y0, w, y1 - y0);
}

void MainWindow::syncControlsFromAnnotations(const ImageAnnotations& ann, const BackendClient::AnalysisPayload& payload) {
    const QSignalBlocker b1(sizeTextEdit_);
    const QSignalBlocker b2(dotTextEdit_);
    const QSignalBlocker b3(dot4TextEdit_);
    const QSignalBlocker b4(sizeVisibleCheck_);
    const QSignalBlocker b5(dotVisibleCheck_);
    const QSignalBlocker b6(dotDateVisibleCheck_);
    const QSignalBlocker b7(sizeCropTopSlider_);
    const QSignalBlocker b8(sizeCropBottomSlider_);
    const QSignalBlocker b9(dotCropTopSlider_);
    const QSignalBlocker b10(dotCropBottomSlider_);
    const QSignalBlocker b11(sizeSectorStartSpin_);
    const QSignalBlocker b12(sizeSectorEndSpin_);
    const QSignalBlocker b13(dotSectorStartSpin_);
    const QSignalBlocker b14(dotSectorEndSpin_);
    const QSignalBlocker b15(wheelCenterXSpin_);
    const QSignalBlocker b16(wheelCenterYSpin_);
    const QSignalBlocker b17(wheelInnerRadiusSpin_);
    const QSignalBlocker b18(wheelOuterRadiusSpin_);
    sizeTextEdit_->setText(ann.sizeText);
    dotTextEdit_->setText(ann.dotText);
    dot4TextEdit_->setText(ann.dot4Text);
    sizeVisibleCheck_->setChecked(ann.sizeVisible);
    dotVisibleCheck_->setChecked(ann.dotVisible);
    dotDateVisibleCheck_->setChecked(ann.dotDateVisible);
    sizeCropTopSlider_->setValue(ann.sizeCropTopPercent);
    sizeCropBottomSlider_->setValue(ann.sizeCropBottomPercent);
    dotCropTopSlider_->setValue(ann.dotCropTopPercent);
    dotCropBottomSlider_->setValue(ann.dotCropBottomPercent);
    sizeSectorStartSpin_->setValue(ann.sizeStartAngleDeg);
    sizeSectorEndSpin_->setValue(ann.sizeEndAngleDeg);
    dotSectorStartSpin_->setValue(ann.dotStartAngleDeg);
    dotSectorEndSpin_->setValue(ann.dotEndAngleDeg);
    wheelCenterXSpin_->setValue(ann.wheelOverrideSet ? ann.wheelCenterX : payload.wheelCenterX);
    wheelCenterYSpin_->setValue(ann.wheelOverrideSet ? ann.wheelCenterY : payload.wheelCenterY);
    wheelInnerRadiusSpin_->setValue(ann.wheelOverrideSet ? ann.wheelInnerRadius : payload.wheelInnerRadius);
    wheelOuterRadiusSpin_->setValue(ann.wheelOverrideSet ? ann.wheelOuterRadius : payload.wheelOuterRadius);
}

QString MainWindow::resolveAnnotationImagePath(const QString& sourceImagePath) const {
    if (sourceImagePath.isEmpty()) {
        return QString();
    }

    const auto payloadIt = payloadsBySourceImage_.constFind(normalizePathKey(sourceImagePath));
    if (payloadIt != payloadsBySourceImage_.constEnd()) {
        if (!payloadIt->unwrappedBandPath.isEmpty() && QFileInfo::exists(payloadIt->unwrappedBandPath)) {
            return payloadIt->unwrappedBandPath;
        }
        for (const auto& image : payloadIt->images) {
            const QString lowerLabel = image.label.toLower();
            const QString lowerPath = image.path.toLower();
            if ((lowerLabel.contains(QStringLiteral("spalla linearizzata")) ||
                 lowerLabel.contains(QStringLiteral("sidewall")) ||
                 lowerLabel.contains(QStringLiteral("unwrap")) ||
                 lowerPath.contains(QStringLiteral("sidewall_band")) ||
                 lowerPath.contains(QStringLiteral("unwrapped")) ||
                 lowerPath.contains(QStringLiteral("polar"))) &&
                QFileInfo::exists(image.path)) {
                return image.path;
            }
        }
    }

    return QString();
}

QString MainWindow::normalizePathKey(const QString& path) const {
    if (path.trimmed().isEmpty()) {
        return QString();
    }
    QString normalized = QFileInfo(path).absoluteFilePath();
    normalized = QDir::cleanPath(normalized);
    normalized.replace('\\', '/');
    return normalized.toLower();
}

QString MainWindow::defaultAnnotationsPathForCurrentSelection() const {
    if (!imagePaths_.isEmpty()) {
        const QFileInfo firstInfo(imagePaths_.first());
        return QDir(firstInfo.absolutePath()).filePath(QStringLiteral(".tyre_reader_annotations.json"));
    }
    return QDir(QDir::currentPath()).filePath(QStringLiteral(".tyre_reader_annotations.json"));
}

QString MainWindow::defaultSuggestionsPathForCurrentSelection() const {
    if (!imagePaths_.isEmpty()) {
        const QFileInfo firstInfo(imagePaths_.first());
        return QDir(firstInfo.absolutePath()).filePath(QStringLiteral("training_set_suggestions/assistant_suggestions.csv"));
    }
    return QString();
}

QString MainWindow::extractImageNumberKey(const QString& path) const {
    const QString fileName = QFileInfo(path).fileName().toLower();
    const QRegularExpression re(QStringLiteral("image[-(]?([0-9]+)"));
    const auto match = re.match(fileName);
    if (!match.hasMatch()) {
        return QString();
    }
    return match.captured(1);
}

void MainWindow::loadSuggestionsFromCsv(const QString& filePath) {
    if (filePath.isEmpty()) {
        return;
    }
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return;
    }

    QTextStream stream(&file);
    if (stream.atEnd()) {
        return;
    }

    const QString headerLine = stream.readLine();
    const QStringList header = parseCsvRow(headerLine);
    QMap<QString, int> columnIndex;
    for (int i = 0; i < header.size(); ++i) {
        columnIndex.insert(header.at(i).trimmed(), i);
    }

    auto idx = [&columnIndex](const QString& name) {
        return columnIndex.value(name, -1);
    };

    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        if (line.trimmed().isEmpty()) {
            continue;
        }
        const QStringList row = parseCsvRow(line);
        const QString imagePath = csvField(row, idx(QStringLiteral("image")));
        QString key = normalizePathKey(imagePath);
        if (key.isEmpty()) {
            continue;
        }

        SuggestionEntry suggestion;
        suggestion.sizeBaseGuess = csvField(row, idx(QStringLiteral("size_base_guess")));
        suggestion.sizeFullGuess = csvField(row, idx(QStringLiteral("size_full_guess")));
        suggestion.sizeConfidence = csvField(row, idx(QStringLiteral("size_confidence")));
        suggestion.sizeLocationHint = csvField(row, idx(QStringLiteral("size_location_hint")));
        suggestion.dotTextGuess = csvField(row, idx(QStringLiteral("dot_text_guess")));
        suggestion.dot4Guess = csvField(row, idx(QStringLiteral("dot4_guess")));
        suggestion.dotConfidence = csvField(row, idx(QStringLiteral("dot_confidence")));
        suggestion.dotLocationHint = csvField(row, idx(QStringLiteral("dot_location_hint")));
        suggestion.reviewStatus = csvField(row, idx(QStringLiteral("review_status")));
        suggestion.notes = csvField(row, idx(QStringLiteral("notes")));
        suggestionsByImage_[key] = suggestion;
    }

    // Fallback match by image number, useful when the same photo exists in another dataset with a different full filename.
    QMap<QString, SuggestionEntry> byNumber;
    for (auto it = suggestionsByImage_.cbegin(); it != suggestionsByImage_.cend(); ++it) {
        const QString numberKey = extractImageNumberKey(it.key());
        if (!numberKey.isEmpty() && !byNumber.contains(numberKey)) {
            byNumber.insert(numberKey, it.value());
        }
    }
    for (const QString& imagePath : imagePaths_) {
        const QString key = normalizePathKey(imagePath);
        if (suggestionsByImage_.contains(key)) {
            continue;
        }
        const QString numberKey = extractImageNumberKey(key);
        if (!numberKey.isEmpty() && byNumber.contains(numberKey)) {
            suggestionsByImage_[key] = byNumber.value(numberKey);
        }
    }
}

void MainWindow::refreshSuggestionPanel() {
    const QString imagePath = currentImagePath();
    const auto it = suggestionsByImage_.constFind(normalizePathKey(imagePath));
    const bool hasSuggestion = imagePath.isEmpty() ? false : it != suggestionsByImage_.constEnd();
    const SuggestionEntry suggestion = hasSuggestion ? it.value() : SuggestionEntry{};

    if (suggestedSizeEdit_ != nullptr) {
        suggestedSizeEdit_->setText(!suggestion.sizeFullGuess.isEmpty() ? suggestion.sizeFullGuess : suggestion.sizeBaseGuess);
    }
    if (suggestedDotEdit_ != nullptr) {
        suggestedDotEdit_->setText(suggestion.dotTextGuess);
    }
    if (suggestedDot4Edit_ != nullptr) {
        suggestedDot4Edit_->setText(suggestion.dot4Guess);
    }
    if (suggestedSizeMetaEdit_ != nullptr) {
        suggestedSizeMetaEdit_->setText(hasSuggestion
                                            ? QStringLiteral("%1 | %2").arg(suggestion.sizeConfidence, suggestion.sizeLocationHint)
                                            : QString());
    }
    if (suggestedDotMetaEdit_ != nullptr) {
        suggestedDotMetaEdit_->setText(hasSuggestion
                                           ? QStringLiteral("%1 | %2").arg(suggestion.dotConfidence, suggestion.dotLocationHint)
                                           : QString());
    }
    if (suggestionNotesView_ != nullptr) {
        suggestionNotesView_->setPlainText(hasSuggestion
                                               ? QStringLiteral("Stato: %1\n%2").arg(suggestion.reviewStatus, suggestion.notes)
                                               : QString());
    }
    if (applySizeSuggestionButton_ != nullptr) {
        applySizeSuggestionButton_->setEnabled(hasSuggestion && (!suggestion.sizeBaseGuess.isEmpty() || !suggestion.sizeFullGuess.isEmpty()));
    }
    if (applyDotSuggestionButton_ != nullptr) {
        applyDotSuggestionButton_->setEnabled(hasSuggestion && (!suggestion.dotTextGuess.isEmpty() || !suggestion.dot4Guess.isEmpty()));
    }
}

void MainWindow::loadAnnotationsFromFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return;
    }
    QJsonParseError parseError;
    const auto doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        return;
    }
    const auto root = doc.object();
    const auto items = root.value(QStringLiteral("items")).toArray();
    for (const auto& value : items) {
        const auto obj = value.toObject();
        const QString key = normalizePathKey(obj.value(QStringLiteral("image")).toString());
        if (key.isEmpty()) {
            continue;
        }
        ImageAnnotations ann;
        ann.sizeStartAngleDeg = obj.value(QStringLiteral("sizeStartAngleDeg")).toDouble();
        ann.sizeEndAngleDeg = obj.value(QStringLiteral("sizeEndAngleDeg")).toDouble();
        ann.sizeSectorSet = obj.value(QStringLiteral("sizeSectorSet")).toBool();
        ann.sizeCropTopPercent = obj.value(QStringLiteral("sizeCropTopPercent")).toInt(0);
        ann.sizeCropBottomPercent = obj.value(QStringLiteral("sizeCropBottomPercent")).toInt(100);
        ann.dotStartAngleDeg = obj.value(QStringLiteral("dotStartAngleDeg")).toDouble();
        ann.dotEndAngleDeg = obj.value(QStringLiteral("dotEndAngleDeg")).toDouble();
        ann.dotSectorSet = obj.value(QStringLiteral("dotSectorSet")).toBool();
        ann.dotCropTopPercent = obj.value(QStringLiteral("dotCropTopPercent")).toInt(0);
        ann.dotCropBottomPercent = obj.value(QStringLiteral("dotCropBottomPercent")).toInt(100);
        ann.sizePreviewPath = obj.value(QStringLiteral("sizePreviewPath")).toString();
        ann.dotPreviewPath = obj.value(QStringLiteral("dotPreviewPath")).toString();
        ann.sizeText = obj.value(QStringLiteral("sizeText")).toString();
        ann.dotText = obj.value(QStringLiteral("dotText")).toString();
        ann.dot4Text = obj.value(QStringLiteral("dot4Text")).toString();
        ann.sizeVisible = obj.value(QStringLiteral("sizeVisible")).toBool(true);
        ann.dotVisible = obj.value(QStringLiteral("dotVisible")).toBool(true);
        ann.dotDateVisible = obj.value(QStringLiteral("dotDateVisible")).toBool(false);
        ann.wheelOverrideSet = obj.value(QStringLiteral("wheelOverrideSet")).toBool(false);
        ann.wheelCenterX = obj.value(QStringLiteral("wheelCenterX")).toDouble();
        ann.wheelCenterY = obj.value(QStringLiteral("wheelCenterY")).toDouble();
        ann.wheelInnerRadius = obj.value(QStringLiteral("wheelInnerRadius")).toDouble();
        ann.wheelOuterRadius = obj.value(QStringLiteral("wheelOuterRadius")).toDouble();
        annotationsByImage_[key] = ann;
    }
}

void MainWindow::saveAnnotationsToFile(const QString& filePath) const {
    QJsonArray items;
    for (auto it = annotationsByImage_.cbegin(); it != annotationsByImage_.cend(); ++it) {
        QJsonObject obj;
        obj.insert(QStringLiteral("image"), it.key());
        obj.insert(QStringLiteral("sizeStartAngleDeg"), it->sizeStartAngleDeg);
        obj.insert(QStringLiteral("sizeEndAngleDeg"), it->sizeEndAngleDeg);
        obj.insert(QStringLiteral("sizeSectorSet"), it->sizeSectorSet);
        obj.insert(QStringLiteral("sizeCropTopPercent"), it->sizeCropTopPercent);
        obj.insert(QStringLiteral("sizeCropBottomPercent"), it->sizeCropBottomPercent);
        obj.insert(QStringLiteral("dotStartAngleDeg"), it->dotStartAngleDeg);
        obj.insert(QStringLiteral("dotEndAngleDeg"), it->dotEndAngleDeg);
        obj.insert(QStringLiteral("dotSectorSet"), it->dotSectorSet);
        obj.insert(QStringLiteral("dotCropTopPercent"), it->dotCropTopPercent);
        obj.insert(QStringLiteral("dotCropBottomPercent"), it->dotCropBottomPercent);
        obj.insert(QStringLiteral("sizePreviewPath"), it->sizePreviewPath);
        obj.insert(QStringLiteral("dotPreviewPath"), it->dotPreviewPath);
        obj.insert(QStringLiteral("sizeText"), it->sizeText);
        obj.insert(QStringLiteral("dotText"), it->dotText);
        obj.insert(QStringLiteral("dot4Text"), it->dot4Text);
        obj.insert(QStringLiteral("sizeVisible"), it->sizeVisible);
        obj.insert(QStringLiteral("dotVisible"), it->dotVisible);
        obj.insert(QStringLiteral("dotDateVisible"), it->dotDateVisible);
        obj.insert(QStringLiteral("wheelOverrideSet"), it->wheelOverrideSet);
        obj.insert(QStringLiteral("wheelCenterX"), it->wheelCenterX);
        obj.insert(QStringLiteral("wheelCenterY"), it->wheelCenterY);
        obj.insert(QStringLiteral("wheelInnerRadius"), it->wheelInnerRadius);
        obj.insert(QStringLiteral("wheelOuterRadius"), it->wheelOuterRadius);
        items.push_back(obj);
    }
    QJsonObject root;
    root.insert(QStringLiteral("items"), items);
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    }
}

void MainWindow::autoSaveAnnotations() const {
    const QString filePath = !annotationsFilePath_.isEmpty() ? annotationsFilePath_ : defaultAnnotationsPathForCurrentSelection();
    if (!filePath.isEmpty()) {
        saveAnnotationsToFile(filePath);
    }
}

QString MainWindow::resolveWheelAnnotationImagePath(const BackendClient::AnalysisPayload& payload) const {
    if (!payload.originalCopyPath.isEmpty() && QFileInfo::exists(payload.originalCopyPath)) {
        return payload.originalCopyPath;
    }
    if (!payload.inputPath.isEmpty() && QFileInfo::exists(payload.inputPath)) {
        return payload.inputPath;
    }
    if (!payload.wheelOverlayPath.isEmpty() && QFileInfo::exists(payload.wheelOverlayPath)) {
        return payload.wheelOverlayPath;
    }
    for (const auto& image : payload.images) {
        const QString lowerLabel = image.label.toLower();
        const QString lowerPath = image.path.toLower();
        if ((lowerPath.contains(QStringLiteral("wheel_annulus")) ||
             lowerPath.contains(QStringLiteral("annulus")) ||
             lowerLabel.contains(QStringLiteral("annulus")) ||
             lowerLabel.contains(QStringLiteral("originale")) ||
             lowerLabel.contains(QStringLiteral("ruota evidenziata")) ||
             lowerLabel.contains(QStringLiteral("wheel")) ||
             lowerPath.contains(QStringLiteral("wheel_contours")) ||
             lowerPath.contains(QStringLiteral("wheel_circles")) ||
             lowerPath.contains(QStringLiteral("wheel.png"))) &&
            QFileInfo::exists(image.path)) {
            return image.path;
        }
    }
    return QString();
}

QString MainWindow::resolveDefaultUnwrapImagePath(const BackendClient::AnalysisPayload& payload) const {
    if (!payload.unwrappedBandPath.isEmpty() && QFileInfo::exists(payload.unwrappedBandPath)) {
        return payload.unwrappedBandPath;
    }
    for (const auto& image : payload.images) {
        const QString lowerLabel = image.label.toLower();
        const QString lowerPath = image.path.toLower();
        if ((lowerLabel.contains(QStringLiteral("spalla linearizzata")) ||
             lowerLabel.contains(QStringLiteral("sidewall")) ||
             lowerLabel.contains(QStringLiteral("unwrap")) ||
             lowerPath.contains(QStringLiteral("sidewall_band")) ||
             lowerPath.contains(QStringLiteral("unwrapped")) ||
             lowerPath.contains(QStringLiteral("polar"))) &&
            QFileInfo::exists(image.path)) {
            return image.path;
        }
    }
    return QString();
}
