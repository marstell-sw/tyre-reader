#include "MainWindow.h"

#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("Tyre Reader Debug GUI"));
    app.setOrganizationName(QStringLiteral("marstell-sw"));

    MainWindow window;
    window.show();
    return app.exec();
}
