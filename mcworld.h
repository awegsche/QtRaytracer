#ifndef MCWORLD_H
#define MCWORLD_H

#include "chunk.h"
#include "nbttag.h"
#include <QMap>
#include <QAbstractItemModel>
#include <vector>


class MCWorld : public QAbstractItemModel
{
public:
    ///<summary>
    /// Chunks nach Chunkposition geordnet. Gehe davon aus, dass das Feld 100*100 regions gross ist,
    /// dann haben wir 3200*3200 chunks. Es handelt sich eh nur um Hashkeys.
    /// </summary>
    //QMap<int, NBTTag*> _chunks;
    std::vector<Chunk*> _chunks;

    int _width;
public:
    MCWorld();

    void addChunk(const int xkey, const int ykey, NBTTag *chunk);
    //Chunk* chunkAt(const int xkey, const int ykey);

    // QAbstractItemModel interface
public:
    QModelIndex index(int row, int column, const QModelIndex &parent) const;
    QModelIndex parent(const QModelIndex &child) const;
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const;
    QVariant data(const QModelIndex &index, int role) const;
};

#endif // WORLD_H
